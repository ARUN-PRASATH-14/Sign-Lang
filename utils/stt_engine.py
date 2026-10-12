"""
Speech-to-Text (STT) Module for Reverse Communication.
Allows normal/hearing individuals to speak into a microphone,
transcribing speech to text for deaf/mute individuals.
"""

import speech_recognition as sr


class STTEngine:
    def __init__(self, energy_threshold: int = 300, pause_threshold: float = 0.8):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = True

    def listen_and_transcribe(self, timeout: int = 5, phrase_time_limit: int = 7):
        """
        Listens to microphone input and converts speech to text.
        Returns tuple: (transcribed_text, status_message, is_success)
        """
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                audio_data = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )

                try:
                    text = self.recognizer.recognize_google(audio_data)
                    return text, "Transcribed successfully!", True
                except sr.UnknownValueError:
                    return "", "Could not understand audio. Please speak clearly.", False
                except sr.RequestError as e:
                    return "", f"Speech API unavailable: {e}", False

        except sr.WaitTimeoutError:
            return "", "Listening timed out. No speech detected.", False
        except Exception as e:
            return "", f"Microphone error: {str(e)}", False


def listen_once(timeout: int = 5):
    engine = STTEngine()
    return engine.listen_and_transcribe(timeout=timeout)
