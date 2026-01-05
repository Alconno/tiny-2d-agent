import threading
from collections import deque
from models.VoiceTranscriber import VoiceTranscriber
from class_models.Context import Context
import time
from core.logging import get_logger
log = get_logger(__name__) 


def build_voice_listener():
    voiceTranscriber = VoiceTranscriber()
    voiceTranscriber._set_asr()
    voiceTranscriber.pause_listener_event = threading.Event()
    voiceTranscriber.pause_listener_event.set()
    context_queue = deque()
    
    def listener():
        while True:
            if voiceTranscriber.pause_listener_event.is_set():
                voice_input = voiceTranscriber()
                if voice_input:
                    #log.info(f"Received voice input: {voice_input}")
                    context_queue.append(Context(voice_input))
            else:
                time.sleep(0.1)

    threading.Thread(target=listener, daemon=True).start()

    return voiceTranscriber, context_queue
