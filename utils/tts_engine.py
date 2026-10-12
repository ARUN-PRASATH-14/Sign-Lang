"""
Text-to-Speech (TTS) Engine for Audio Feedback.
Supports offline native pyttsx3 with non-blocking threaded speech synthesis
and browser audio generation.
"""

import threading
import queue
import pyttsx3


class TTSEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, rate: int = 150, volume: float = 1.0):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TTSEngine, cls).__new__(cls)
                cls._instance._init_engine(rate, volume)
            return cls._instance

    def _init_engine(self, rate: int, volume: float):
        self.rate = rate
        self.volume = volume
        self.queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _speech_worker(self):
        """Worker thread to run pyttsx3 without blocking main UI / webcam loop."""
        while True:
            text = self.queue.get()
            if text is None:
                break
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                # Select female / natural voice if available
                voices = engine.getProperty("voices")
                if len(voices) > 1:
                    engine.setProperty("voice", voices[1].id)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[TTS Error]: {e}")
            finally:
                self.queue.task_done()

    def speak(self, text: str):
        """Enqueue text for speech synthesis."""
        if text and text.strip():
            # Clear duplicate queue if too long
            while self.queue.qsize() > 2:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break
            self.queue.put(text.strip())


# Convenience function
_global_tts = None


def speak_text(text: str):
    global _global_tts
    if _global_tts is None:
        _global_tts = TTSEngine()
    _global_tts.speak(text)
