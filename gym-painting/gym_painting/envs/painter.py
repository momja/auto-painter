import cv2
import numpy as np
from skimage.draw import line

class Painter:
    def paint_from_states(self, states, canvas=None, width=None, height=None, channels=3):
        """


        """

        assert canvas is not None or (width is not None and height is not None)

        if canvas is None:
            canvas = np.zeros((height, width, channels))


        if len(states) < 2:
            return canvas

        prev_state = states[0]
        for state in states[1:]:
            if not state["pendown"] or not prev_state["pendown"]:
                prev_state = state
                continue
            img = self._line_interp(canvas,
                                prev_state["pos"],
                                state["pos"],
                                prev_state["color"],
                                state["color"],
                                prev_state["radius"],
                                state["radius"])
            prev_state = state
        return canvas


    def _line_interp(self, img, start, end, c1, c2, r1=5, r2=5):
        line_iter = list(zip(*line(*start, *end)))
        img = img.copy()
        for i, pt in enumerate(line_iter):
            alpha = i / len(line_iter)
            radius = int(r1 * (1.0 - alpha) + r2 * alpha)
            color_at_pt = c1 * (1.0 - alpha) + c2 * alpha
            #img[pt[0],pt[1]] = c1 * (1.0 - alpha) + c2 * alpha
            img = cv2.circle(img, pt, radius, color_at_pt, -1)
        return img


if __name__ == "__main__":
    state1 = State()
    state1.pendown = 1
    state1.color = np.array([1,1,1])
    state1.x = 10
    state1.y = 10
    state1.radius = 3
    state2 = State()
    state2.pendown = 1
    state2.color = np.array([1,1,0])
    state2.x = 89
    state2.y = 89
    state2.radius = 6
    state3 = State()
    state3.pendown = 1
    state3.color = np.array([1,0,0])
    state3.x = 50
    state3.y = 10
    state3.radius = 10


    img = create_image([state1, state2, state3], 100, 100)
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
