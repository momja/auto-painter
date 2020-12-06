from tkinter import *
from PIL import ImageTk, Image
from multiprocessing import Process,Pipe
import numpy as np

class Renderer:
    def __init__(self):
        self.r_fd, self.w_fd = Pipe(False)
        self.p = Process(target=Renderer._render_image_server, args=(self.r_fd,))


    def start_server(self):
        self.p.start()


    def close_server(self):
        self.r_fd.close
        self.p.join()

    
    def update_render(self, template, canvas):
        # Combine template and canvas to one image
        combined_img = np.hstack([template, canvas])
        print(combined_img.shape)
        self.w_fd.send(combined_img)


    @staticmethod
    def _render_image_server(fd_reader):
        root = Tk()
        root.resizable(True, True)
        try:
            img_np = fd_reader.recv()   # Blocking
        except EOFError:
            # Parent process has ended, or write side of pipe is closed
            sys.exit(0)
        img = ImageTk.PhotoImage(Image.fromarray(img_np))
        panel = Label(root, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")

        Renderer._update_render(fd_reader, panel, root)
        root.mainloop()


    @staticmethod
    def _update_render(r_fd, panel, root):
        # update view
        print("waiting for update")
        print("new image received")
        try:
            img = r_fd.recv()
        except EOFError:
            # Parent process has ended, or write side of pipe is closed
            sys.exit(0)
        img_update = ImageTk.PhotoImage(Image.fromarray(img))
        panel.configure(image=img_update)
        panel.image = img_update
        root.after(100, Renderer._update_render, r_fd, panel, root)


#if __name__ == "__main__":
#    start_render_server()
