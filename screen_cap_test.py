import tkinter as tk
from PIL import ImageGrab

# Global variables
start_x = start_y = end_x = end_y = 0
selection_done = False

def on_mouse_down(event):
    global start_x, start_y
    start_x, start_y = event.x, event.y

def on_mouse_move(event):
    canvas.delete("rect")
    canvas.create_rectangle(start_x, start_y, event.x, event.y, outline="red", width=2, tag="rect")

def on_mouse_up(event):
    global end_x, end_y, selection_done
    end_x, end_y = event.x, event.y
    selection_done = True
    root.quit()

def select_region():
    global root, canvas, selection_done
    selection_done = False

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.3)
    root.attributes("-topmost", True)
    canvas = tk.Canvas(root, cursor="cross")
    canvas.pack(fill="both", expand=True)

    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)

    root.mainloop()

    if selection_done:
        # Sort coordinates so left < right, top < bottom
        x1, x2 = sorted([start_x, end_x])
        y1, y2 = sorted([start_y, end_y])

        screenshot = ImageGrab.grab()
        cropped = screenshot.crop((x1, y1, x2, y2))
        cropped.show()

if __name__ == "__main__":
    print("Run the script, click and drag to select a region of the screen.")
    select_region()
