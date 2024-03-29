# -*- coding: utf-8 -*-
# ************************************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
# ************************************************************

import math

from ..utils import distance, rotate_around, normal_dist, uniform_angle
from .base import BaseDevice

from aitk.utils import Color


BLUE = Color("blue", 64)
BLACK = Color("black", 64)
WHITE = Color("white", 64)

class Compass(BaseDevice):
    def __init__(self, position=(0, 0), name="compass", **kwargs):
        """
        A compass for detecting direction to beacon.

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
        return "<Compass %r position=%r>" % (self.name, self.position)

    def initialize(self):
        """
        Internal method to set all settings to default values.
        """
        self.type = "compass"
        self.name = "compass"
        self.value = 0.0
        self.values = [0, 0, 0, 0]
        self.length = 30  # CM
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
        self.values = [0, 0, 0, 0]
        # Location of sensor:
        p = rotate_around(
            self.robot.x,
            self.robot.y,
            self.dist_from_center,
            self.robot.a + self.dir_from_center + math.pi / 2,
        )
        beacon = self.robot.world.beacon
        if beacon is not None and beacon.state == "on":
            x, y = beacon.get_position(world=True)
            to_beacon = uniform_angle(math.pi * 3/2 - math.atan2(p[0] - x, p[1] - y))
            front = uniform_angle(self.robot.a)
            self.value = uniform_angle(to_beacon - front)

            segment = 2 * math.pi / 4.0
            MAX = math.pi * 2
            # 0: front, 1: counter-clockwise, left, 2: back, 3: right
            if (0 < self.value < segment/2) or ((MAX - segment/2) < self.value < MAX):
                self.values[0] = 1
            elif (1 * segment/2 < self.value < 3 * segment/2):
                self.values[1] = 1
            elif (3 * segment/2 < self.value < 5 * segment/2):
                self.values[2] = 1
            else:
                self.values[3] = 1

    def draw(self, backend):
        """
        Draw the device on the backend.

        Args:
            backend (Backend): an aitk drawing backend
        """
        def get_arc(quadrant):
            if quadrant == 0: # front
                start = - 1 * math.pi / 4
                stop = math.pi / 4
            elif quadrant == 1: # left
                start = 1 * math.pi / 4
                stop = 3 * math.pi / 4
            elif quadrant == 2: # back
                start = -3 * math.pi / 4
                stop = -5 * math.pi / 4
            elif quadrant == 3: # right
                start =  - 3 * math.pi / 4
                stop = - 1 * math.pi / 4
            return start, stop

        for quadrant in range(4):
            start, stop = get_arc(quadrant)
            if self.values[quadrant]:
                backend.lineWidth(1)
                backend.set_stroke_style(BLACK)
                backend.set_fill_style(BLUE)
            else:
                backend.lineWidth(1)
                backend.set_stroke_style(BLACK)
                backend.set_fill_style(WHITE)
            backend.draw_arc(0, 0, self.length, self.length, start, stop)

    def watch(self, title="Compass:"):
        """
        Create a dynamically updating view
        of this sensor.

        Args:
            title (str): title of sensor
        """
        widget = self.get_widget(title=title)
        return display(widget)

    def get_value(self):
        """
        Get the direction to the beacon. The value
        is in radians, with 0 pointing straight ahead of the
        robot, and increasing counter-clockwise back
        to 2 pi.
        """
        return self.value

    def get_values(self):
        """
        Get the values of the compass as an array of zeros
        and a single 1 in the quadrant indicating where the beacon
        is relative to the robot.

        The indices of the array are as shown relative to the
        robot:

        \  0  /
         \   /
          \ /
        1     3
          / \
         /   \
        /  2  \

        where 0=front, 1=left, 2=back, and 3=right.

        So that the meaning of the array is:

        [front, left, back, right]
        """
        return self.values

    def get_widget(self, title="Compass:"):
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
                self, "name", "value", "values", title=title, labels=["Name:", "Compass:", "Values:"]
            )
            self.robot.world._watchers.append(self._watcher)

        return self._watcher.widget

    def set_position(self, position):
        """
        Set the position of the compass with respect to the center of the
        robot.

        Args:
            position (List[int, int]): represents [x, y] in CM from
                center of robot
        """
        if len(position) != 2:
            raise ValueError("position must be of length two")

        self.position = position
        # Get location of sensor, doesn't change once position is set:
        self.dist_from_center = distance(0, 0, self.position[0], self.position[1])
        self.dir_from_center = math.atan2(-self.position[0], self.position[1])
