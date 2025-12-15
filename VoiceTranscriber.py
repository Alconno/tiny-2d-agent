from pynput import keyboard
import sounddevice as sd, numpy as np, tempfile
from scipy.io.wavfile import write
from transformers import pipeline
import re

class VoiceTranscriber:
    def __init__(self, sr=16000, ch=1, model="openai/whisper-small", listener="on"):
        self.sr = sr
        self.ch = ch
        self.buf = []
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            return_timestamps=False,
            generate_kwargs={"task": "transcribe", "language": "en"}
        )
        self.pressed = False
        self.listener_enabled = listener.lower() == "on"
        
        if self.listener_enabled:
            self.listener = keyboard.Listener(on_press=self._press, on_release=self._release)
            self.listener.start()

    def _press(self, key):
        if key == keyboard.Key.f8:
            self.pressed = True

    def _release(self, key):
        if key == keyboard.Key.f8:
            self.pressed = False

    def _cb(self, indata, frames, t, status):
        self.buf.append(indata.copy())

    def __call__(self):
        while not self.pressed:
            sd.sleep(1)
        self.buf = []
        with sd.InputStream(samplerate=self.sr, channels=self.ch, callback=self._cb):
            while self.pressed:
                sd.sleep(1)
        audio = (np.concatenate(self.buf, 0) * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            write(f.name, self.sr, audio)
            p = f.name
        
        return self.asr(p)["text"] if self.listener_enabled else None
