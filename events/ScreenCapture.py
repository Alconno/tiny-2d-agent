from enum import Flag, auto
import tkinter as tk

class ScreenCaptureEvent(Flag):
    CAPTURE = auto()

class ScreenCapture:
    def __init__(self):
        self.box = [0, 0, 0, 0]
        self.selection_done = False

    def on_mouse_down(self, event):
        self.box[0:2] = event.x, event.y

    def on_mouse_move(self, event):
        self.canvas.delete("rect")
        self.canvas.create_rectangle(*self.box[0:2], event.x, event.y,
                                     outline="red", width=2, tag="rect")

    def on_mouse_up(self, event):
        self.box[2:4] = event.x, event.y
        self.selection_done = True
        self.root.quit()

    def select_region(self):
        self.selection_done = False

        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True, "-alpha", 0.3, "-topmost", True)
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.root.mainloop()
        self.root.destroy()

        if self.selection_done:
            x1, y1 = self.box[0], self.box[1]
            x2, y2 = self.box[2], self.box[3]

            left, right = sorted([x1, x2])
            top, bottom = sorted([y1, y2])

            left = max(left, 0)
            top = max(top, 0)

            bbox_xywh = (left, top, right - left, bottom - top)
            bbox_coords = (left, top, right, bottom)

            return bbox_xywh, bbox_coords
        return None, None

