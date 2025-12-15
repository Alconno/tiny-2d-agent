import zmq
import pickle
import OCR as OCR_module
from OCR import OCR
from finetune.Qwen3_06BInsctruct.generate import generate, setup_gpt_model
from embed import embed_text, get_emb_model
import logging, os
import traceback
logging.basicConfig(level=logging.ERROR)

def main():
    logging.getLogger("ppocr").setLevel(logging.WARNING)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    ocr = OCR(conf=0.4, downscale=1.0, max_workers=8, upscale=2.0, box_condense=(8,4))

    # ZMQ setup
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")

    gpt_model, gpt_tokenizer, get_prompt_func = setup_gpt_model()

    emb_model = get_emb_model()

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print("ZMQ ready, models are being hosted.\n")

    try:
        while True:
            socks = dict(poller.poll(timeout=1000))
            if socket in socks and socks[socket] == zmq.POLLIN:
                try:
                    req = pickle.loads(socket.recv(zmq.NOBLOCK))
                    cmd = req.get("cmd")

                    if cmd == "ocr":
                        preds = ocr(req["img"], req["ocr_crop_offset"])
                        socket.send(pickle.dumps(preds))

                    elif cmd == "embed":
                        emb = embed_text(req["text"], emb_model)
                        socket.send(pickle.dumps(emb))

                    elif cmd == "gpt":
                        out = generate(req["input"], gpt_model, gpt_tokenizer, get_prompt_func)
                        socket.send(pickle.dumps(out))

                    else:
                        socket.send(pickle.dumps({"error": f"Unknown cmd {cmd}"}))

                except Exception as e:
                    logging.exception("Server exception")
                    tb = traceback.format_exc()
                    try:
                        socket.send(pickle.dumps({
                            "error": str(e),
                            "traceback": tb
                        }))
                    except zmq.error.ZMQError:
                        pass

    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
