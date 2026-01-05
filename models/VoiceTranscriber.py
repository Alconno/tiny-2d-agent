from pynput import keyboard
import sounddevice as sd, numpy as np, tempfile
from scipy.io.wavfile import write
from transformers import pipeline
from core.logging import get_logger
log = get_logger(__name__) 

class VoiceTranscriber:
    def __init__(self, sr=16000, ch=1, model="openai/whisper-small", listener="on"):
        self.sr = sr
        self.ch = ch
        self.buf = []
        self.model = model
        self.asr = None
        self.pressed = False
        self.listener_enabled = listener.lower() == "on"
        self.pause_listener_event = None
        
        if self.listener_enabled:
            self.listener = keyboard.Listener(on_press=self._press, on_release=self._release)
            self.listener.start()

    def _set_asr(self):
        if self.asr is None:
            log.info("> Setting up ASR")
            self.asr = pipeline(
                "automatic-speech-recognition", # asr
                model=self.model,
                return_timestamps=False,
                generate_kwargs={"task": "transcribe", "language": "en"}
            )

    def _press(self, key):
        if key == keyboard.Key.f8:
            self.pressed = True

    def _release(self, key):
        if key == keyboard.Key.f8:
            self.pressed = False

    def _cb(self, indata, frames, t, status):
        self.buf.append(indata.copy())

    def __call__(self):
        if not self.asr:
            log.debug("You didnt set asr with 'voiceTranscriber._set_asr()' - Make sure its in main thread")
            return None
        
        while not self.pressed:
            sd.sleep(1)

        self.buf = []
        with sd.InputStream(samplerate=self.sr, channels=self.ch, callback=self._cb):
            while self.pressed:
                sd.sleep(1)
        
        if not self.buf: 
            log.warning("No audio captured")
            return None
        
        audio = (np.concatenate(self.buf, 0) * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            write(f.name, self.sr, audio)
            p = f.name
        
        return self.asr(p)["text"] if self.listener_enabled else None