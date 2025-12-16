import zmq
import pickle

class AccessModels():
    def __init__(self):
        TCP = "tcp://127.0.0.1:5555"
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(TCP)
        self.socket.RCVTIMEO = 45000  # recv timeout
        self.socket.SNDTIMEO = 45000  # send timeout
        print("Socket connected to", TCP)

    def safe_request(self, msg):
        try:
            self.socket.send(pickle.dumps(msg))
            resp = self.socket.recv()
            return pickle.loads(resp)

        except zmq.error.Again:
            print("ZMQ timeout — resetting socket")

            # HARD RESET (required)
            self.socket.close(linger=0)
            context = zmq.Context.instance()
            self.socket = context.socket(zmq.REQ)
            self.socket.connect("tcp://127.0.0.1:5555")
            self.socket.RCVTIMEO = 45000
            self.socket.SNDTIMEO = 45000

        return None

    def ocr_func(self, img, ocr_crop_offset):
        return self.safe_request({"cmd": "ocr", "img": img, "ocr_crop_offset": ocr_crop_offset})

    def embd_func(self, txt):
        return self.safe_request({"cmd": "embed", "text": txt})

    def gpt_func(self, input_text):
        return self.safe_request({"cmd": "gpt", "input": input_text})
