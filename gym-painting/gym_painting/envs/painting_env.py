import logging
from collections import OrderedDict
import math
from math import pi
import random
import os

from gym_painting.envs.rendering import Renderer

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import cv2

from gym_painting.envs.painter import Painter

logger = logging.getLogger(__name__)


OBS_FRAME_SHAPE = (31, 31, 3)  # Area around the current position that the user can view
EPISODE_SIZE = 5000


class PaintingEnv(gym.Env):
    """
    Environment for training an agent to paint by mimicking painting brushstrokes.

    """

    metadata = {"render.modes": ["rgb_array", "human"]}

    def __init__(self):
        self.__version__ = "0.1.0"
        self.viewer = None
        self.template = None
        self.canvas = None
        self.cur_state = {}
        self.state_history = []
        self.action_history = []
        self.painter = Painter()
        self.renderer = None
        self._configure_environment()
        logger.info(f"PaintingEnv - Version {self.__version__}")

        self.cur_step = -1

        # -- ACTION SPACE -- #
        # ------------------ #

        color_space = spaces.Box(
            np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1])
        )  # (hue, saturation, value)
        motion_space = spaces.Box(
            np.array([-math.pi, 0, -3]), np.array([math.pi, 20, 3])
        )  # (direction, distance, radius)
        brush_space = spaces.Discrete(2)  # (pen up, pen down)
        self.action_space = spaces.Dict(
            {"color": color_space, "motion": motion_space, "pendown": brush_space}
        )

        # -- OBSERVATION SPACE -- #
        # ----------------------- #

        img_patch_space = spaces.Box(low=0, high=1, shape=OBS_FRAME_SHAPE)
        pen_down = spaces.Discrete(2)
        motion_space = spaces.Box(np.array([0, 1]), np.array([2 * math.pi, 10]))
        color_space = spaces.Box(np.array([0, 0, 0]), np.array([1, 1, 1]))

        self.observation_space = spaces.Dict(
            {
                "patch": img_patch_space,
                "color": color_space,
                "motion": motion_space,
                "pendown": brush_space,
            }
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _configure_environment(self):
        """
        Configure Painting and starting position for brush.
        Can override for custom implementation.

        """
        self._start_painter()

    def _start_painter(self, painting_name="the_starry_night_sm.jpg"):
        """
        Given a painting filepath, create a new painting server

        """

        assert OBS_FRAME_SHAPE[0] % 2 == 1 and OBS_FRAME_SHAPE[1] % 2 == 1

        painting_fpath = os.path.join(
            os.path.dirname(__file__), "samples", painting_name
        )

        self.template_rgb = cv2.imread(painting_fpath)
        # Convert to HSV Color space
        self.template = cv2.cvtColor(self.template_rgb, cv2.COLOR_BGR2HSV)
        # Convenience for render function
        self.template_rgb = cv2.cvtColor(self.template_rgb, cv2.COLOR_BGR2RGB)
        # Normalize template
        self.template = self.template.astype(np.float) / 255
        pad_h, pad_w = OBS_FRAME_SHAPE[0] // 2, OBS_FRAME_SHAPE[1] // 2
        self.template = np.pad(
            self.template, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), "constant"
        )
        self.canvas = np.zeros_like(self.template_rgb, dtype=np.float)

    def _get_obs(self):
        """
        Returns current observation
        - image patch around brush position
        - brush radius
        - previous brush direction
        - previous color
        """
        x, y = self.cur_state["pos"]
        obs = np.zeros((spaces.flatdim(self.observation_space)))

        obs = np.hstack((
            self._get_template_patch(x, y).flatten(),
            self.cur_state["motion"],
            self.cur_state["color"],
            self.cur_state["pendown"]
        ))
        return np.asarray(obs)

    def _get_template_patch(self, x, y):
        return self.template[y : y + OBS_FRAME_SHAPE[1], x : x + OBS_FRAME_SHAPE[0]]

    def _get_canvas_patch(self, x, y):
        start_y, start_x = y - OBS_FRAME_SHAPE[1]//2, y - OBS_FRAME_SHAPE[0]//2
        end_y, end_x = y + OBS_FRAME_SHAPE[1]//2 + 1, y + OBS_FRAME_SHAPE[0]//2 + 1
        return self.canvas[start_y:end_y, start_x:end_x]

    def step(self, action):
        """
        Take action on current state

        """

        assert self.cur_step <= EPISODE_SIZE

        # compute new state
        self._take_action(action)

        # update canvas with new state
        self._update_canvas(start_state=-1)

        # compute reward
        reward = self._get_reward()

        self.cur_step += 1

        # terminate if all pixels in canvas are filled or
        # the current step has exceed the steps in one episode
        terminal_state = (
            self.cur_step > EPISODE_SIZE or self.canvas[self.canvas == 0].size == 0
        )

        if (terminal_state):
            self.renderer.close_server()

        return self._get_obs(), reward, terminal_state, {}

    def _compute_gradient(img):
        """
        Computes the gradient of an image with a specified filter size. Returns gradients in the X and Y direction
        """

    def _get_reward(self):
        """
        Given the current state, compute the reward.
        Reward is given based on color, radius, and trajectory matching
        with the true painting

        """
        x, y = self.cur_state["pos"]
        return 1/(np.linalg.norm(self._get_template_patch(x, y) - self._get_canvas_patch(x, y)))

    def _unflatten_action(self, flat_action):
        return OrderedDict(
            ("color", flat_action[:4]),
            ("motion", flat_action[4:7]),
            ("pendown", round(flat_action[7]))
        )

    def _take_action(self, action):
        """"""

        action = self._unflatten_action(action)

        max_y, max_x = self.template_rgb.shape[0] - 1, self.template_rgb.shape[1] - 1
        direction, distance, radius = action["motion"]
        prev_dir = self.cur_state["motion"][0]
        pos_update = np.array(
            [np.cos(direction + prev_dir), np.sin(direction + prev_dir)]
        )
        pos_update *= distance  # Multiply by distance
        pos_update = pos_update.astype(int)

        new_direction = self.cur_state["motion"][0] + action["motion"][0]
        new_direction = new_direction % (math.pi * 2)
        new_radius = np.clip(self.cur_state["motion"][1] + action["motion"][1], 1, 10)

        prev_state = self.cur_state
        new_state = {
            "pos": np.clip(prev_state["pos"] + pos_update, [0, 0], [max_x, max_y]),
            "motion": np.array([new_direction, new_radius]),
            "color": np.clip(prev_state["color"] + action["color"], 0, 1),
            "pendown": prev_state["pendown"] ^ action["pendown"],
        }
        self.cur_state = new_state
        self.state_history.append(prev_state)
        self.action_history.append(action)

        return self.cur_state

    def reset(self):
        # randomly set position, radius, etc.
        max_y, max_x = self.template.shape[0:2]
        start_pos = np.array([random.randrange(0, max_x), random.randrange(0, max_y)])
        self.canvas = np.zeros_like(self.template_rgb)
        self.cur_state = {
            "pos": start_pos,
            "motion": np.array([0, 1]),
            "color": self.template[start_pos[1], start_pos[0]],
            "pendown": 0,
        }
        self.action_history = []
        self.state_history = []
        self.cur_step = -1
        return self._get_obs()

    def _start_render_process(self):
        """
        Start server on renderer.
        We can then update the renderer by providing the current
        canvas and template.

        """
        self.renderer = Renderer()
        self.renderer.start_server()

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            return self.canvas
        elif mode == "human":
            if not self.renderer:
                self._start_render_process()
            self.renderer.update_render(
                self.template_rgb, (self.canvas * 255).astype(np.uint8), tuple(self.cur_state["pos"])
            )

    def _update_canvas(self, start_state=0, end_state=None):
        """
        Given state history, generate the canvas that results from interpolating
        between each state

        Parameters
        ----------
        start_state : int, optional
            the state index to start updating from. Default is 0
        end_state : int, optional
            the state index to end updating at. Default is None, or last index

        Returns
        -------
        numpy.ndarray
            numpy array of same shape as template representing the updated canvas

        """
        if not end_state:
            self.canvas = self.painter.paint_from_states(
                self.state_history, shape=self.canvas.shape
            )
        else:
            self.canvas = self.painter.paint_from_states(
                self.state_history, shape=self.canvas.shape
            )
        return self.canvas

    def close(self):
        if self.renderer:
            self.renderer.close_server()
