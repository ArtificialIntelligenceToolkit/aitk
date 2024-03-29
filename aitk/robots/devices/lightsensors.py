# -*- coding: utf-8 -*-
# ************************************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
# ************************************************************

import math

from ..utils import distance, rotate_around, normal_dist
from .base import BaseDevice

from aitk.utils import Color

PURPLE = Color("purple")
YELLOW = Color("yellow")
BLACK = Color("black")

class LightSensor(BaseDevice):
    def __init__(self, position=(0, 0), name="light",
                 color_sensitivity=None, **kwargs):
        """
        A light sensor for sensing bulbs.

        Args:
            position (Tuple[int, int]): the position of the
                sensor in centimeters relative to the center
                of the robot.
            name (str): the name of the device
            color_sensitivity (Color): the name of the color
                that the sensor is sensitive to. None means
                that it is sensitive to all colors.
        """
        config = {
            "position": position,
            "name": name,
            "color_sensitivity": color_sensitivity,
        }
        config.update(kwargs)
        self._watcher = None
        self.robot = None
        self.initialize()
        self.from_json(config)

    def __repr__(self):
        return "<LightSensor %r position=%r color_sensitivity=%r>" % (
            self.name, self.position, self.color_sensitivity)

    def initialize(self):
        """
        Internal method to set all settings to default values.
        """
        self.type = "light"
        self.name = "light"
        self.color_sensitivity = None
        self.value = 0.0
        # FIXME: add to config
        self.multiplier = 1000  # CM
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
            "position", "name", "class", "color_sensitivity"
        ])
        self.verify_config(valid_keys, config)

        if "name" in config:
            self.name = config["name"]
        if "color_sensitivity" in config:
            if config["color_sensitivity"] is not None:
                self.color_sensitivity = Color(config["color_sensitivity"])
            else:
                self.color_sensitivity = None
            config["color_sensitivity"] = self.color_sensitivity
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
            "color_sensitivity": str(self.color_sensitivity) if self.color_sensitivity is not None else None,
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
        for bulb in self.robot.world._get_light_sources(all=True):  # for each light source:
            if bulb.robot is self.robot:
                # You can't sense your own bulbs
                continue

            if ((self.color_sensitivity is not None) and
                (self.color_sensitivity != bulb.color)):
                # If sensitivity is set, if bulb doesn't match
                # skip it
                continue

            z, brightness, light_color = (  # noqa: F841
                bulb.z,
                bulb.brightness,
                bulb.color,
            )
            x, y = bulb.get_position(world=True)

            angle = math.atan2(x - p[0], y - p[1])
            dist = distance(x, y, p[0], p[1])
            ignore_robots = []
            if bulb.robot is not None:
                ignore_robots.append(bulb.robot)
            if self.robot is not None:
                ignore_robots.append(self.robot)

            hits = self.robot.cast_ray(p[0], p[1], angle, dist, ignore_robots=ignore_robots)
            if self.robot.world.debug and draw_list is not None:
                draw_list.append(("draw_circle", (p[0], p[1], 2), {}))
                draw_list.append(("draw_circle", (x, y, 2), {}))

                for hit in hits:
                    draw_list.append(("set_fill_style", (PURPLE,), {}))
                    draw_list.append(("draw_circle", (hit.x, hit.y, 2), {}))

            if len(hits) == 0:  # nothing blocking! we can see the light
                # Maximum value of 100.0 with defaults:
                self.value += (normal_dist(dist, 0, brightness) / math.pi) / brightness
                self.value = min(self.value, 1.0)
                if draw_list is not None:
                    draw_list.append(("strokeStyle", (PURPLE, 1), {}))
                    draw_list.append(("draw_line", (x, y, p[0], p[1]), {}))

    def draw(self, backend):
        """
        Draw the device on the backend.

        Args:
            backend (Backend): an aitk drawing backend
        """
        backend.lineWidth(1)
        backend.set_stroke_style(BLACK)
        if self.color_sensitivity is not None:
            backend.set_fill_style(self.color_sensitivity)
        else:
            backend.set_fill_style(YELLOW)
        backend.draw_circle(self.position[0], self.position[1], 2)

    def get_brightness(self):
        """
        Get the light brightness reading from the sensor.
        """
        return self.value

    def watch(self, title="Light Sensor:"):
        """
        Create a dynamically updating view
        of this sensor.

        Args:
            title (str): title of sensor
        """
        widget = self.get_widget(title=title)
        return display(widget)

    def get_widget(self, title="Light Sensor:"):
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
                self, "name", "value", title=title, labels=["Name:", "Light:"]
            )
            self.robot.world._watchers.append(self._watcher)

        return self._watcher.widget

    def set_position(self, position):
        """
        Set the position of the light sensor with respect to the center of the
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
