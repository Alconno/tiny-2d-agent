import threading
from collections import deque
from models.VoiceTranscriber import VoiceTranscriber
from class_models.Context import Context
import time
from core.logging import get_logger
log = get_logger(__name__) 


def build_voice_listener():
    voice_transcriber = VoiceTranscriber()
    voice_transcriber.pause_listener_event = threading.Event()
    voice_transcriber.pause_listener_event.set()
    context_queue = deque()
    
    def listener():
        while True:
            if voice_transcriber.pause_listener_event.is_set():
                voice_input = voice_transcriber()
                if voice_input:
                    log.debug("Received voice input: ", voice_input)
                    context_queue.append(Context(voice_input))
            else:
                time.sleep(0.1)

    threading.Thread(target=listener, daemon=True).start()

    return voice_transcriber, context_queue
