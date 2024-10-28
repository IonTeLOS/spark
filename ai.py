import sys
import os
import re
import json
import subprocess

import google.generativeai as genai
from google.generativeai import GenerativeModel
import speech_recognition as sr
from langdetect import detect
from PySide6.QtCore import Qt, QDateTime, Signal, QObject, QTimer, QUrl, QMetaObject, Q_ARG
from PySide6.QtGui import QFontDatabase, QColor, QPalette, QTextCursor, QIcon, QTextOption, QTextCharFormat, QTextCursor, QFont, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QFileDialog,
    QLineEdit, QLabel, QDockWidget, QHBoxLayout, QMessageBox, QScrollBar,
    QStyle, QProgressBar, QSlider, QComboBox, QCheckBox
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
import requests
from gtts import gTTS
import pygame
import threading
import time
from functools import partial
import logging
import qtawesome as qta
from qt_material import apply_stylesheet

# Set up basic logging configuration to save logs to a specific file
logging.basicConfig(
    filename="gemini_app.log",  # Log file name
    level=logging.DEBUG,         # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

class AudioController(QObject):
    audio_finished = Signal()
    audio_error = Signal(str)
    audio_ready = Signal(str)

    def __init__(self):
        super().__init__()
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        
        self.audio_output.setVolume(1.0)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)
        self.player.errorOccurred.connect(self.on_error)
        self.player.playbackStateChanged.connect(self.on_playback_state_changed)

    def play_audio(self, audio_path):
        try:
            url = QUrl.fromLocalFile(os.path.abspath(audio_path))
            self.player.setSource(url) 
            print(f"Attempting to play audio: {audio_path}")
            self.player.play()  
        except Exception as e:
            self.audio_error.emit(str(e))
            print(f"Error starting playback: {e}")

    def on_playback_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            print("Playback started.")
        elif state == QMediaPlayer.StoppedState:
            print("Playback stopped.")
            self.audio_finished.emit() 

    def on_media_status_changed(self, status):
        print(f"Media status changed: {status}")
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            print("Playback finished")
            self.audio_finished.emit()
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            print("Invalid media detected")
            self.audio_error.emit("Invalid media.")

    def on_error(self, error):
        error_msg = self.player.errorString()
        print(f"Playback error: {error_msg}")
        self.audio_error.emit(error_msg)

    def stop_audio(self):
        print("Stopping audio playback")
        self.player.stop()
        self.audio_finished.emit()

class LoadingSpinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.spinner = QProgressBar()
        self.spinner.setRange(0, 0)  
        self.spinner.setTextVisible(False)
        self.spinner.setMaximumHeight(2)
        
        self.label = QLabel("Converting to audio...")
        self.label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.label)
        layout.addWidget(self.spinner)
        
        self.setMaximumHeight(50)
        self.hide()

class AudioControlWidget(QWidget):
    def __init__(self, audio_controller, parent=None):
        super().__init__(parent)
        self.audio_controller = audio_controller
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(qta.icon("mdi.stop", scale_factor=1.5))
        self.stop_btn.clicked.connect(self.audio_controller.stop_audio)
        layout.addWidget(self.stop_btn)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50) 
        self.volume_slider.valueChanged.connect(self.set_volume)
        layout.addWidget(self.volume_slider)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        self.audio_controller.player.positionChanged.connect(self.update_progress)
        self.audio_controller.player.durationChanged.connect(self.update_duration)
        
        self.duration = 0
        
        self.hide()
        
    def set_volume(self, value):
        self.audio_controller.audio_output.setVolume(value / 100.0)
        
    def update_duration(self, duration):
        self.duration = duration
        self.progress_bar.setValue(0)
        
    def update_progress(self, position):
        if self.duration > 0:
            progress = int((position / self.duration) * 100)
            self.progress_bar.setValue(progress)

class GeminiApp(QWidget):
    response_received = Signal(str)
    play_audio_signal = Signal(str) 
    show_controls_signal = Signal(bool)     
    speech_recognized = Signal(str)

    def __init__(self, as_dockable=False):
        super().__init__()
        self.as_dockable = as_dockable
        self.api_key = None
        self.conversations = []
        self.available_models = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
        self.default_model = "gemini-1.5-flash"
        self.current_model = self.load_model_selection()
        
        self.setWindowTitle("Spark")
        self.setGeometry(300, 300, 500, 600)
        
        self.audio_controller = AudioController()
        self.audio_controls = AudioControlWidget(self.audio_controller)
        self.audio_controls.setVisible(False)
        self.play_audio_signal.connect(self.audio_controller.play_audio)
        self.show_controls_signal.connect(self.audio_controls.setVisible)
        self.speech_recognized.connect(self.handle_recognized_speech)
        self.audio_controller.audio_finished.connect(self.on_audio_finished)
        self.audio_controller.audio_error.connect(self.handle_audio_error)
        self.current_audio_path = None
        self.is_audio_playing = False
        self.font_size = self.load_font_size() or 14
        self.setup_ui()
        self.font_size_slider.setValue(self.font_size)
        self.update_font_size(self.font_size) 
        self.load_api_key()
        if not self.check_internet():
            self.display_message("System", "No internet connection. Please check your network.")

        self.response_received.connect(self.handle_response)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        palette = self.palette()
        self.text_color = palette.color(QPalette.Text)
        self.base_color = palette.color(QPalette.Base)
       
        self.api_key_label = QLabel('<a href="https://aistudio.google.com/app/apikey">Click here to get your API key<br></a>')
        self.api_key_label.setToolTip(
            "Entering your personal API key (free of charge) is necessary to use this app. \nClick here to open https://aistudio.google.com/app/apikey in order to obtain your API key."
        )
        self.api_key_label.setAlignment(Qt.AlignRight) 
        self.api_key_label.setOpenExternalLinks(True)
        layout.addWidget(self.api_key_label)
       
        self.model_label = QLabel("Select AI Model:")
        self.model_dropdown = QComboBox()
        self.model_dropdown.setToolTip(
            "Select the AI model to interact with."
        )
        self.model_dropdown.addItems(self.available_models)
        
        index = self.available_models.index(self.current_model)
        self.model_dropdown.setCurrentIndex(index)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_dropdown)
        
        self.model_dropdown.currentIndexChanged.connect(self.update_selected_model)
        self.api_key_input = QLineEdit(self)
        self.api_key_input.setToolTip(
            "To use Gemini paste here the API key you got from Google."
        )
        self.api_key_input.setPlaceholderText("Enter the API key you got from Google")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.api_key_input)

        self.set_api_key_btn = QPushButton()
        self.set_api_key_btn.setToolTip(
            "Click to set your API key to the key you entered above. \nIt will be stored locally for future use."
        )
        self.set_api_key_btn.setIcon(qta.icon("mdi.key-plus", scale_factor=1.5)) 
        self.set_api_key_btn.clicked.connect(self.set_api_key)
        layout.addWidget(self.set_api_key_btn)

        self.change_api_key_btn = QPushButton()
        self.change_api_key_btn.setToolTip(
            "Click to add or change your API key. \nA valid API key obtained from Google is necessary to use Gemini."
        )
        self.change_api_key_btn.setIcon(qta.icon("mdi.key-change", scale_factor=1.5))  
        self.change_api_key_btn.clicked.connect(self.show_api_input)
        layout.addWidget(self.change_api_key_btn)
        
        search_layout = QHBoxLayout()

        self.search_input = QLineEdit(self)
        self.search_input.setToolTip(
            "Type something to search for. \nUse the arrow buttons to navigate to previous or next occurrences."
        )
        self.search_input.setPlaceholderText("Search chat...")
        self.search_input.textChanged.connect(self.search_text)
        search_layout.addWidget(self.search_input)

        self.prev_button = QPushButton(self)
        self.prev_button.setIcon(qta.icon("mdi.arrow-left", scale_factor=1.5))
        self.prev_button.clicked.connect(self.highlight_previous_match)
        search_layout.addWidget(self.prev_button)

        self.next_button = QPushButton(self)
        self.next_button.setIcon(qta.icon("mdi.arrow-right", scale_factor=1.5))
        self.next_button.clicked.connect(self.highlight_next_match)
        search_layout.addWidget(self.next_button)

        layout.addLayout(search_layout)
        
        self.chat_display = QTextEdit(self)
        self.chat_display.setToolTip(
            "This is your chat history with Gemini."
        )
        self.chat_display.setLineWrapMode(QTextEdit.WidgetWidth)
        self.chat_display.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_display.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.chat_display.setReadOnly(True)
        self.chat_display.setTextColor(self.text_color)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
            }
        """)
        layout.addWidget(self.chat_display)

        self.chat_display.append("Welcome to Spark AI Chat! Powered by Gemini. \nAccording to Google's Terms of Use you must be at least 18 years old to use this service.")
        
        self.font_size_slider = QSlider(Qt.Horizontal)
        self.font_size_slider.setToolTip(
            "Slide to either direction to set preferred text size."
        )
        self.font_size_slider.setRange(10, 26)  
        self.font_size_slider.setValue(self.font_size) 
        self.font_size_slider.valueChanged.connect(self.update_font_size)
        layout.addWidget(self.font_size_slider)

        self.spinner = LoadingSpinner()
        layout.addWidget(self.spinner)
        self.audio_controls = AudioControlWidget(self.audio_controller)
        layout.addWidget(self.audio_controls)
        
        input_layout = QHBoxLayout()

        self.context_checkbox = QCheckBox("")
        self.context_checkbox.setChecked(True)
        self.context_checkbox.setToolTip(
            "If checked, includes all previous chat messages as context for your next prompt. \nIf unchecked your prompt will be submitted without context."
        )
        input_layout.addWidget(self.context_checkbox)

        self.message_input = QLineEdit(self)
        self.message_input.setToolTip(
            "Enter your prompt here to start or continue your discussion with Gemini."
        )
        self.message_input.setFixedHeight(40)
        self.message_input.setClearButtonEnabled(True)
        self.message_input.setStyleSheet("font-size: 18px;")
        self.message_input.setPlaceholderText("Ask me anything..")
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)

        # Create the send button with an icon
        self.send_btn = QPushButton()
        self.send_btn.setToolTip(
            "Click here to submit your message to Gemini."
        )
        self.send_btn.setIcon(qta.icon("mdi.send", scale_factor=1.5))
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        # Add the input layout to the main layout
        layout.addLayout(input_layout)


        btn_layout = QHBoxLayout()
        self.file_btn = QPushButton()
        self.file_btn.setToolTip(
            "Choose a file to submit to Gemini."
        )
        self.file_btn.setIcon(qta.icon("mdi.file-upload", scale_factor=1.5))  
        self.file_btn.clicked.connect(self.send_file)
        btn_layout.addWidget(self.file_btn)

        self.speech_btn = QPushButton()
        self.speech_btn.setToolTip(
            "Click here to speak to Gemini."
        )
        self.speech_btn.setIcon(qta.icon("mdi.microphone", scale_factor=1.5)) 
        self.speech_btn.clicked.connect(self.send_speech)
        btn_layout.addWidget(self.speech_btn)

        self.read_btn = QPushButton()
        self.read_btn.setToolTip(
            "Read aloud Gemini's last response."
        )
        self.read_btn.setIcon(qta.icon("mdi.volume-high", scale_factor=1.5)) 
        self.read_btn.clicked.connect(self.read_last_response)
        btn_layout.addWidget(self.read_btn)
        layout.addLayout(btn_layout)

        # Create an HBoxLayout for the delete and open logs buttons
        convo_layout = QHBoxLayout()

        # Delete All button
        self.delete_all_btn = QPushButton()
        self.delete_all_btn.setToolTip(
            "Delete all previous messages and responses from Gemini.\n"
            "An archive file will be stored in your file system for future reference."
        )
        self.delete_all_btn.setIcon(qta.icon("mdi.delete-forever", scale_factor=1.5))
        self.delete_all_btn.clicked.connect(self.delete_all_conversations)
        convo_layout.addWidget(self.delete_all_btn)

        # Open Logs button
        self.open_logs_btn = QPushButton()
        self.open_logs_btn.setToolTip("Open the location where conversation archives are stored.")
        self.open_logs_btn.setIcon(qta.icon("mdi.folder-open", scale_factor=1.5))
        self.open_logs_btn.clicked.connect(self.open_logs_location)
        convo_layout.addWidget(self.open_logs_btn)

        # Add the button layout to the main layout
        layout.addLayout(convo_layout)

        self.setLayout(layout)
        self.load_conversations()
    
    def open_logs_location(self):
        logs_directory = os.path.abspath(".")  # Adjust path if you store archives in a specific folder
        QDesktopServices.openUrl(QUrl.fromLocalFile(logs_directory))  
    
    def search_text(self):
        search_term = self.search_input.text().strip()

        # Clear highlights if search input is empty
        if not search_term:
            self.clear_highlights()
            self.scroll_to_bottom()
            return

        # Find and highlight matches
        self.matches = self.find_all_matches(search_term)
        self.current_match_index = -1  # Reset to start

        # Highlight the first match
        self.highlight_next_match()

    def find_all_matches(self, search_term):
        """Find all matches of the search term and return their positions."""
        matches = []
        cursor = self.chat_display.textCursor()
        cursor.setPosition(0)
        
        # Set format for highlighted text
        highlight_format = QTextCharFormat()
        highlight_format.setBackground(QColor("yellow"))

        while True:
            cursor = self.chat_display.document().find(search_term, cursor)
            if cursor.isNull():
                break
            matches.append(cursor)
            # Highlight each match
            cursor.mergeCharFormat(highlight_format)

        return matches

    def clear_highlights(self):
        """Clear all highlights in the QTextEdit."""
        cursor = self.chat_display.textCursor()
        cursor.select(QTextCursor.Document)
        
        # Reset format to default
        default_format = QTextCharFormat()
        cursor.mergeCharFormat(default_format)
        
        # Reset cursor
        self.chat_display.setTextCursor(cursor)

    def highlight_next_match(self):
        """Highlight the next match in the list."""
        if not self.matches:
            return

        # Move to the next match index
        self.current_match_index = (self.current_match_index + 1) % len(self.matches)
        self.scroll_to_match(self.matches[self.current_match_index])

    def highlight_previous_match(self):
        """Highlight the previous match in the list."""
        if not self.matches:
            return

        # Move to the previous match index
        self.current_match_index = (self.current_match_index - 1) % len(self.matches)
        self.scroll_to_match(self.matches[self.current_match_index])

    def scroll_to_match(self, cursor):
        """Scroll to a specific QTextCursor position."""
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def scroll_to_bottom(self):
        """Scroll to the bottom of the QTextEdit."""
        self.chat_display.moveCursor(QTextCursor.End)
        
    def update_font_size(self, value):
        # Create a QTextCharFormat with the desired font size
        format = QTextCharFormat()
        format.setFontPointSize(value)
        
        # Select all existing text and apply only the font size, preserving HTML
        cursor = self.chat_display.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.mergeCharFormat(format)

        # Clear selection and move cursor to the end
        cursor.clearSelection()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)


    def save_font_size(self, value):
        # Save the font size to the config file
        config = {"font_size": value}
        with open("", "w") as file:
            json.dump(config, file)

    def load_font_size(self):
        # Load the font size from the config file
        if os.path.exists("config.json"):
            with open("config.json", "r") as file:
                config = json.load(file)
                return config.get("font_size", 14)  # Default to 14 if not found
        return 14  # Default font size if config file doesn't exist

          
    def update_selected_model(self):
        self.current_model = self.model_dropdown.currentText()
        self.save_model_selection()

    def save_model_selection(self):
        with open("config.json", "w") as f:
            json.dump({"selected_model": self.current_model}, f)

    def load_model_selection(self):
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                config = json.load(f)
                return config.get("selected_model", self.default_model)
        return self.default_model
    
    def load_api_key(self):
        if os.path.exists("api_key.txt"):
            with open("api_key.txt", "r") as f:
                self.api_key = f.read().strip()
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                    self.api_key_input.hide()
                    self.set_api_key_btn.hide()
                    self.change_api_key_btn.show()
                    self.display_message("System", "API key loaded successfully.")

    def show_api_input(self):
        self.api_key_input.show()
        self.set_api_key_btn.show()
        self.change_api_key_btn.hide()

    def set_api_key(self):
        self.api_key = self.api_key_input.text().strip()
        if self.api_key:
            genai.configure(api_key=self.api_key)
            with open("api_key.txt", "w") as f:
                f.write(self.api_key)
            self.api_key_input.hide()
            self.set_api_key_btn.hide()
            self.change_api_key_btn.show()
            self.display_message("System", "API key set and saved.")

    def check_internet(self):
        try:
            requests.get("https://www.google.com", timeout=3)
            return True
        except requests.ConnectionError:
            return False

    def send_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self.display_message("Me", content)
                        response = self.get_gemini_response(content)
                        self.display_message("Gemini", response)
                        self.save_conversation(content, response)
            except Exception as e:
                self.display_message("System", f"Error reading file: {e}")

    def send_speech(self):
        self.display_message("System", "Listening...")
        threading.Thread(target=self.recognize_speech_background, daemon=True).start()

    def recognize_speech_background(self):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                audio = recognizer.listen(source) 

            message = recognizer.recognize_google(audio)
           
            self.speech_recognized.emit(message)

        except sr.UnknownValueError:
            self.speech_recognized.emit("Could not understand audio.")
        except sr.RequestError:
            self.speech_recognized.emit("Could not request results; check your network.")

    def handle_recognized_speech(self, message):
        if message:
            self.display_message("Me", message)
            response = self.get_gemini_response(message)
      
            self.display_message("Gemini", response)
            self.save_conversation(message, response)
            
    def send_message(self):
        message = self.message_input.text().strip()
        if not message:
            self.display_message("System", "Cannot send an empty message.")
            return

        # If the "Context" checkbox is checked, prepend previous messages
        if self.context_checkbox.isChecked():
            # Get the full text from chat_display and format it as context
            previous_messages = self.chat_display.toPlainText().strip()
            context_message = f"Context for my prompt: {previous_messages}\n\n{message}"
        else:
            context_message = message
            
        self.display_message("Me", message)
        self.message_input.clear()

        target_function = partial(self.process_message, context_message)

        threading.Thread(target=target_function, daemon=True).start()

    def process_message(self, message):
        try:
            logging.debug(f"Processing message: {message}")
            QTimer.singleShot(0, lambda: self.spinner.show())

            response = self.get_gemini_response(message)
            logging.debug(f"Received response: {response}")

            self.response_received.emit(response)
            self.save_conversation(message, response)
            QTimer.singleShot(0, lambda: self.spinner.hide())
        except Exception as e:
            QTimer.singleShot(0, lambda: self.display_message("System", f"Error: {str(e)}"))
            logging.error(f"Error in process_message: {e}")

    def get_gemini_response(self, message):
        """
        Get a response from the Gemini API
        
        Args:
            message (str): The user's input message
        
        Returns:
            str: Gemini's response text or an error message
        """
        if not self.api_key:
            return "API key is missing. Please set it first."

        try:
            model = GenerativeModel(self.current_model)
            response = model.generate_content(message)
            
            if hasattr(response, 'text'):
                return response.text.strip()
            elif isinstance(response, list) and response:
                return response[0].text.strip()
            else:
                return "No valid response received from Gemini. Please try again."
                
        except Exception as e:
            logging.error(f"Error while fetching response: {e}")
            return f"Error while fetching response: {str(e)}"

    def handle_response(self, response):
        self.display_message("Gemini", response)

    def display_message(self, sender, message, timestamp=None):
        try:
            logging.debug(f"display_message called with sender: {sender}, message: {message}, timestamp: {timestamp}")

            # Get the current font size from the slider
            font_size = self.font_size_slider.value()

            if not timestamp:
                timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")

            # Check if the message is a code block and format accordingly
            if self.is_code_block(message):
                formatted_message = (
                    f"<pre style='font-family: JetBrains Mono; font-size: {font_size}px;'><code>"
                    f"{self.escape_html(message.strip('```'))}"
                    f"</code></pre>"
                )
            else:
                # Escape HTML and apply formatting for regular messages
                formatted_message = message.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')

            # Define the color for each sender type
            sender_color = {
                "System": "#808080",  # Gray for system messages
                "Me": "#2196F3",      # Blue for user messages
                "Gemini": "#4CAF50"   # Green for Gemini responses
            }.get(sender, "#000000")  # Default black color if sender not found

            # Construct HTML with dynamic font size
            html = f"""
                <div style='margin-bottom: 8px; font-size: {font_size}px;'>
                    <span style='color: #666666; font-size: 0.9em;'>{timestamp}</span><br>
                    <span style='color: {sender_color}; font-weight: bold;'>{sender}:</span>
                    <span style='color: #ffffff;'>{formatted_message}</span>
                </div>
                <div style='border-bottom: 1px solid #e0e0e0; margin: 8px 0;'></div>
            """

            # Insert the constructed HTML at the end of the QTextEdit content
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertHtml(html)
            self.scroll_to_bottom()

        except Exception as e:
            logging.error(f"Error in display_message: {e}")
            self.chat_display.append(f"Error displaying message: {e}\n")

    
    def is_code_block(self, message):
        return (
            re.search(r'```|`[^`]+`', message) or            # Triple or inline backticks
            re.search(r'\b(def|class|import|for|while)\b', message) or  # Common code keywords
            re.search(r' {4}|\t', message)                    # Indentation of 4 spaces or tab
        )

    def escape_html(self, text):
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    def start_audio_playback(self, audio_path):
        self.font_size_slider.hide()
        self.show_controls_signal.emit(True)
        QMetaObject.invokeMethod(self.audio_controls, "setVisible", Qt.QueuedConnection, Q_ARG(bool, True))
        QMetaObject.invokeMethod(self.audio_controls.progress_bar, "setValue", Qt.QueuedConnection, Q_ARG(int, 0))
        self.play_audio_signal.emit(audio_path)
        QMetaObject.invokeMethod(self.audio_controller, "play_audio", Qt.QueuedConnection, Q_ARG(str, audio_path))
        self.spinner.hide()
        self.is_audio_playing = True

    def on_audio_finished(self):
        self.is_audio_playing = False
        self.show_controls_signal.emit(False)
        self.audio_controls.hide()
        self.font_size_slider.show()

    def handle_audio_error(self, error_msg):
        self.display_message("System", f"Audio error: {error_msg}")
        self.is_audio_playing = False
        self.spinner.hide()
        self.audio_controls.hide()

    def read_last_response(self):
        if self.is_audio_playing:
            self.audio_controller.stop_audio()
            return

        last_response = self.get_last_gemini_response()
        if not last_response:
            self.display_message("System", "No response to read.")
            return

        self.spinner.show()

        def generate_audio():
            try:
                if not os.path.exists("audio"):
                    os.makedirs("audio")
               
                detected_language = detect(last_response)
                print(f"Detected language: {detected_language}")
                timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
                audio_path = os.path.join("audio", f"AI-response_{timestamp}.mp3")
                tts = gTTS(text=last_response, lang=detected_language)

                tts.save(audio_path)
                
                self.audio_controller.audio_ready.emit(audio_path)
                
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    self.current_audio_path = audio_path
                    QTimer.singleShot(200, lambda: self.start_audio_playback(audio_path))
                else:
                    print("File save error: Audio file could not be found or is empty.")
                    self.handle_audio_error("File save error: Audio file could not be found or is empty.")

            except Exception as e:
                QTimer.singleShot(0, lambda: self.handle_audio_error(str(e)))
                
        self.audio_controller.audio_ready.connect(self.start_audio_playback)
        threading.Thread(target=generate_audio, daemon=True).start()

    def save_conversation(self, user_input, response):
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        self.conversations.append({"timestamp": timestamp, "input": user_input, "output": response})
        with open("conversations.json", "w") as f:
            json.dump(self.conversations, f, indent=2)

    def load_conversations(self):
        if os.path.exists("conversations.json"):
            with open("conversations.json", "r") as f:
                self.conversations = json.load(f)
            for conv in self.conversations:
                timestamp = conv.get("timestamp", "")
                self.display_message("Me", conv['input'], timestamp)
                self.display_message("Gemini", conv['output'], timestamp)

    def delete_all_conversations(self):
        reply = QMessageBox.question(self, "Delete All", "Are you sure you want to delete all conversations?", 
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Archive the current conversation to a .txt file before deleting
            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
            archive_filename = f"archived_conversation_{timestamp}.txt"

            # Retrieve all text from chat_display
            with open(archive_filename, "w") as archive_file:
                archive_file.write(self.chat_display.toPlainText())

            # Clear conversations in memory and in UI
            self.conversations = []
            with open("conversations.json", "w") as f:
                json.dump(self.conversations, f)
            self.chat_display.clear()
            self.display_message("System", "All conversations deleted and archived.")


    def scroll_to_bottom(self):
        scroll_bar = self.chat_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def get_dock_widget(self):
        dock = QDockWidget("Gemini", self)
        dock.setWidget(self)
        return dock

    def get_last_gemini_response(self):
        if self.conversations:
            return self.conversations[-1]["output"]
        return None

def ensure_config_files():
    if not os.path.exists("config.json"):
        default_config = {"font_size": 14, "selected_model": "gemini-1.5-flash"}
        with open("config.json", "w") as config_file:
            json.dump(default_config, config_file)
 
    if not os.path.exists("api_key.txt"):
        with open("api_key.txt", "w") as api_file:
            api_file.write("")
            
def main(as_dockable=False):
    app = QApplication(sys.argv)
    app.setApplicationName("Spark")
    apply_stylesheet(app, theme='dark_teal.xml')
    app.setFont(QFont("Roboto", 14)) 

    current_version = "1.0.0" 
    app_location = sys.executable
    update_url = "https://marko-app.netlify.app/ai.json"  

    # Path to the updater executable
    updater_path = os.path.join(os.path.dirname(sys.executable), 'sum')
    
    if os.path.exists(updater_path):
        try:
            subprocess.run(
                [
                    updater_path,
                    "--current_version", current_version,
                    "--current_location", app_location,
                    "--url", update_url,
                    "--app-name", "Spark",
                    "--interactive"
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Updater process failed: {e}")
    else:
        print(f"Updater not found at {updater_path}")

    
    # Initialize the main application window
    chat_app = GeminiApp(as_dockable)
    if getattr(sys, 'frozen', False):  # Check if the app is frozen (e.g., by PyInstaller)
        icon_path = os.path.join(sys._MEIPASS, "spark.svg")
    else:
        icon_path = os.path.join(os.path.abspath("."), "spark.svg")      
    app.setWindowIcon(QIcon(icon_path))
    
    if as_dockable:
        main_window = QWidget()
        main_window.setWindowTitle("Main Application with Gemini Dock")
        layout = QVBoxLayout(main_window)
        dock = chat_app.get_dock_widget()
        layout.addWidget(dock)
        main_window.setLayout(layout)
        main_window.show()
    else:
        chat_app.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    ensure_config_files()
    main()
