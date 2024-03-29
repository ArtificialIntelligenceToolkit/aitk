# -*- coding: utf-8 -*-
# ************************************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
# ************************************************************

import math

from ..utils import distance, rotate_around
from .base import BaseDevice

from aitk.utils import Color

PURPLE = Color("purple")
WHITE = Color("white")
BLACK = Color("black")

class SmellSensor(BaseDevice):
    def __init__(self, position=(0, 0), name="smell", **kwargs):
        """
        A smell sensor for sensing food.

        Args:
            position (Tuple[int, int]): the position of the
                sensor in centimeters relative to the center
                of the robot.
            name (str): the name of the device
        """
        config = {
            "position": position,
            "name": name,
        }
        config.update(kwargs)
        self._watcher = None
        self.robot = None
        self.initialize()
        self.from_json(config)

    def __repr__(self):
        return "<SmellSensor %r position=%r>" % (self.name, self.position,)

    def initialize(self):
        """
        Internal method to set all settings to default values.
        """
        self.type = "smell"
        self.name = "smell"
        self.value = 0.0
        self.position = [0, 0]
        self.dist_from_center = distance(0, 0, self.position[0], self.position[1])
        self.dir_from_center = math.atan2(-self.position[0], self.position[1])

    def from_json(self, config):
        """
        Set the settings from a device config.

        Args:
            config (dict): a config dictionary
        """
        valid_keys = set([
            "position", "name", "class"
        ])
        self.verify_config(valid_keys, config)

        if "name" in config:
            self.name = config["name"]
        if "position" in config:
            self.position = config["position"]
            # Get location of sensor, doesn't change once position is set:
            self.dist_from_center = distance(0, 0, self.position[0], self.position[1])
            self.dir_from_center = math.atan2(-self.position[0], self.position[1])

    def to_json(self):
        """
        Save the internal settings to a config dictionary.
        """
        config = {
            "class": self.__class__.__name__,
            "position": self.position,
            "name": self.name,
        }
        return config

    def _step(self, time_step):
        pass

    def update(self, draw_list=None):
        """
        Update the device.

        Args:
            draw_list (list): optional. If given, then the
                method can add to it for drawing later.
        """
        self.value = 0
        # Location of sensor:
        p = rotate_around(
            self.robot.x,
            self.robot.y,
            self.dist_from_center,
            self.robot.a + self.dir_from_center + math.pi / 2,
        )
        self.value = self.robot.world._grid.get(p[0], p[1])

    def draw(self, backend):
        """
        Draw the device on the backend.

        Args:
            backend (Backend): an aitk drawing backend
        """
        backend.lineWidth(1)
        backend.set_stroke_style(BLACK)
        backend.set_fill_style(WHITE)
        backend.draw_circle(self.position[0], self.position[1], 2)

    def get_reading(self):
        """
        Get the smell reading from the sensor.
        """
        return self.value

    def watch(self, title="Smell Sensor:"):
        """
        Create a dynamically updating view
        of this sensor.

        Args:
            title (str): title of sensor
        """
        widget = self.get_widget(title=title)
        return display(widget)

    def get_widget(self, title="Smell Sensor:"):
        """
        Return the dynamically updating widget.

        Args:
            title (str): title of sensor
        """
        from ..watchers import AttributesWatcher

        if self.robot is None or self.robot.world is None:
            print("ERROR: can't watch until added to robot, and robot is in world")
            return None

        if self._watcher is None:
            self._watcher = AttributesWatcher(
                self, "name", "value", title=title, labels=["Name:", "Food:"]
            )
            self.robot.world._watchers.append(self._watcher)

        return self._watcher.widget

    def set_position(self, position):
        """
        Set the position of the sensor with respect to the center of the
        robot.

        Args:
            position (List[int, int]) represents [x, y] in CM from
                center of robot
        """
        if len(position) != 2:
            raise ValueError("position must be of length two")

        self.position = position
        # Get location of sensor, doesn't change once position is set:
        self.dist_from_center = distance(0, 0, self.position[0], self.position[1])
        self.dir_from_center = math.atan2(-self.position[0], self.position[1])
