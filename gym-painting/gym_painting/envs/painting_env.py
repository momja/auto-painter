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
EPISODE_SIZE = 10000


class PaintingEnv(gym.Env):
    """
    Environment for training an agent to paint by mimicking painting brushstrokes.

    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self):
        self.__version__ = "0.1.0"
        self.viewer = None
        self.template = None
        self.canvas = None
        self.cur_state = {}
        self.state_history = []
        self.action_history = []
        self.painter = Painter()
        self.renderer = Renderer()
        self._configure_environment()
        self._start_render_process()
        logger.info(f"PaintingEnv - Version {self.__version__}")

        self.cur_step = -1

        self.seed()
        self.reset()

        # -- ACTION SPACE -- #
        color_space = spaces.Box(
            np.array([0, 0, 0]), np.array([1, 1, 1])
        )  # (hue, saturation, value)
        motion_space = spaces.Box(
            np.array([-math.pi, 0, 0]), np.array([math.pi, 20, 3])
        )  # (direction, distance, radius)
        brush_space = spaces.MultiBinary(1)  # (pen up, pen down)
        self.action_space = spaces.Dict(
            {"color": color_space, "motion": motion_space, "pendown": brush_space}
        )

        # -- OBSERVATION SPACE -- #
        img_patch_space = spaces.Box(low=0, high=1, shape=OBS_FRAME_SHAPE)
        brush_space = spaces.MultiBinary(1)
        motion_space = spaces.Box(np.array([0, 0]), np.array([2 * math.pi, 50]))
        color_space = spaces.Box(np.array([0, 0, 0]), np.array([1, 1, 1]))

        self.observation_space = spaces.Dict(
            {
                "patch": img_patch_space,
                "color": color_space,
                "motion": motion_space,
                "pendown": brush_space,
            }
        )

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

        self.template = cv2.imread(painting_fpath)
        self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
        pad_h, pad_w = OBS_FRAME_SHAPE[0] // 2, OBS_FRAME_SHAPE[1] // 2
        self.template = np.pad(
            self.template, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), "constant"
        )
        self.canvas = np.zeros_like(self.template)

    def _get_obs(self):
        """
        Returns current observation
        - image patch around brush position
        - brush radius
        - previous brush direction
        - previous color
        """
        x, y = self.cur_state["pos"]

        obs = OrderedDict()

        obs["patch"] = self._get_template_patch(x, y)
        obs["motion"] = self.cur_state["motion"]
        obs["color"] = self.cur_state["color"]
        obs["pendown"] = self.cur_state["pendown"]
        return obs

    def _get_template_patch(self, x, y):
        print(x,y)
        return self.template[y : y + OBS_FRAME_SHAPE[1], x : x + OBS_FRAME_SHAPE[0]]

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
        # import code
        # code.interact(local=locals())
        return self._get_obs(), reward, terminal_state, {}

    def _get_reward(self):
        """
        Given the current state, compute the reward.
        Reward is given based on color, radius, and trajectory matching
        with the true painting

        """
        return 0

    def _take_action(self, action):
        """"""

        print(action)
        max_y, max_x = self.template.shape[0] - 1, self.template.shape[1] - 1
        pos_update = np.array(
            [np.cos(action["motion"][0]), np.sin(action["motion"][1])]
        )
        pos_update *= action["motion"][2]  # Multiply by distance
        pos_update = pos_update.astype(int)

        prev_state = self.cur_state
        new_state = {}
        new_state = {
            "pos": np.clip(prev_state["pos"] + pos_update, [0, 0], [max_x, max_y]),
            "motion": prev_state["motion"] + action["motion"][[0, 2]],
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
        self.canvas = np.zeros_like(self.template)
        self.cur_state = {
            "pos": start_pos,
            "motion": np.array([0, 5]),
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
        if self.renderer:
            self.renderer.start_server()

    def render(self, mode="rgb_array"):
        if self.renderer:
            self.renderer.update_render(self.template, self.canvas)

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
            self.painter.paint_from_states(
                self.state_history[start_state:], self.canvas
            )
        else:
            self.painter.paint_from_states(
                self.state_history[start_state : end_state + 1], self.canvas
            )

        return self.canvas

    def close(self):
        if self.renderer:
            self.renderer.close_server()
