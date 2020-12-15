import cv2
import numpy as np
from skimage.draw import line

class Painter:
    def paint_from_states(self, states, canvas=None, shape=None):
        """


        """

        assert canvas is not None or shape is not None
        
        if canvas is None:
            height, width, channels = shape
            canvas = np.zeros((height, width, channels), dtype=np.float)

        if len(states) < 2:
            return canvas

        prev_state = states[0]

        for state in states[1:]:
            if not state["motion"][2] or not prev_state["motion"][2]:
                prev_state = state
                continue
            canvas = self._line_interp(canvas,
                                prev_state["pos"],
                                state["pos"],
                                prev_state["color"],
                                state["color"],
                                prev_state["motion"][1],
                                state["motion"][1])
            prev_state = state
        return canvas


    def _line_interp(self, img, start, end, c1, c2, r1=5, r2=5):
        """
        
        """

        line_iter = list(zip(*line(*start, *end)))
        for i, pt in enumerate(line_iter):
            alpha = i / len(line_iter)
            radius = int(r1 * (1.0 - alpha) + r2 * alpha)
            color_at_pt = c1 * (1.0 - alpha) + c2 * alpha
            #img[pt[0],pt[1]] = c1 * (1.0 - alpha) + c2 * alpha
            img = cv2.circle(img, pt, radius, color_at_pt, -1)
        return img
