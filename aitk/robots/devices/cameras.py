# -*- coding: utf-8 -*-
# ************************************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
# ************************************************************

import math

from ..utils import (
    distance,
    display,
    PI_OVER_180,
    PI_OVER_2,
    TWO_PI,
    ONE80_OVER_PI,
    rotate_around,
    uniform_angle,
)
from .base import BaseDevice

from aitk.utils import Color

WHITE = Color("white")

class Camera(BaseDevice):
    def __init__(
        self,
        width=64,
        height=32,
        fov=30,
        colorsFadeWithDistance=1.0,
        sizeFadeWithDistance=0.8,
        reflectGround=True,
        reflectSky=False,
        max_range=1000,
        name="camera",
        samples=1,
        **kwargs
    ):
        """
        A camera device.

        Args:
            width (int): width of camera in pixels
            height (int): height of camera in pixels
            fov (float): width of camera field of view in degrees. Can be
                180 or even 360 for wide angle cameras.
            colorsFadeWithDistance (float): colors get darker
                faster with larger value
            sizeFadeWithDistance (float): size gets smaller faster
                with with larger value
            reflectGround (bool): ground reflects for 3D point cloud
            reflectSky (bool): sky reflects for 3D point cloud
            max_range (int): maximum range of camera
            name (str): the name of the camera
            samples (int): how many pixels should it sample

        Note: currently the camera faces forward.
        """
        config = {
            "width": width,
            "height": height,
            "fov": fov,
            "colorsFadeWithDistance": colorsFadeWithDistance,
            "sizeFadeWithDistance": sizeFadeWithDistance,
            "reflectGround": reflectGround,
            "reflectSky": reflectSky,
            "max_range": max_range,
            "name": name,
            "samples": samples,
        }
        config.update(kwargs)
        self._watcher = None
        self.robot = None
        self.initialize()
        self.from_json(config)

    def initialize(self):
        """
        Internal method to set all settings to default values.
        """
        # FIXME: camera is fixed at (0,0) facing forward
        self.type = "camera"
        self.bulbs_are_visible = True
        self.food_is_visible = True
        self.time = 0.0
        self.cameraShape = [64, 32]
        self.position = [0, 0]
        self.max_range = 1000
        self.samples = 1
        self.name = "camera"
        # 0 = no fade, 1.0 = max fade
        self.colorsFadeWithDistance = 0.9
        self.sizeFadeWithDistance = 1.0
        self.reflectGround = True
        self.reflectSky = False
        self.set_fov(60)  # degrees
        self.reset()

    def reset(self):
        """
        Internal method to reset the camera data.
        """
        self.hits = [[] for i in range(self.cameraShape[0])]

    def from_json(self, config):
        """
        Set the settings from a device config.

        Args:
            config (dict): a config dictionary
        """
        valid_keys = set([
            "width", "name", "height", "colorsFadeWithDistance",
            "sizeFadeWithDistance", "reflectGround", "reflectSky",
            "fov", "max_range", "samples", "position", "class"
        ])
        self.verify_config(valid_keys, config)

        if "width" in config:
            self.cameraShape[0] = config["width"]
        if "height" in config:
            self.cameraShape[1] = config["height"]

        if "colorsFadeWithDistance" in config:
            self.colorsFadeWithDistance = config["colorsFadeWithDistance"]
        if "sizeFadeWithDistance" in config:
            self.sizeFadeWithDistance = config["sizeFadeWithDistance"]
        if "reflectGround" in config:
            self.reflectGround = config["reflectGround"]
        if "reflectSky" in config:
            self.reflectSky = config["reflectSky"]
        if "fov" in config:
            self.set_fov(config["fov"])  # degrees
        if "max_range" in config:
            self.max_range = config["max_range"]
        if "samples" in config:
            self.samples = config["samples"]
        if "name" in config:
            self.name = config["name"]
        if "position" in config:
            self.position = config["position"]

    def to_json(self):
        """
        Save the internal settings to a config dictionary.
        """
        return {
            "class": self.__class__.__name__,
            "width": self.cameraShape[0],
            "height": self.cameraShape[1],
            "colorsFadeWithDistance": self.colorsFadeWithDistance,
            "sizeFadeWithDistance": self.sizeFadeWithDistance,
            "reflectGround": self.reflectGround,
            "reflectSky": self.reflectSky,
            "fov": self.fov * ONE80_OVER_PI,  # save in degrees
            "max_range": self.max_range,
            "samples": self.samples,
            "name": self.name,
            "position": self.position,
        }

    def __repr__(self):
        return "<Camera %r size=(%r,%r), fov=%r>" % (
            self.name,
            self.cameraShape[0],
            self.cameraShape[1],
            round(self.fov * ONE80_OVER_PI, 2),
        )

    def watch(self, **kwargs):
        """
        Create a dynamically updating view
        of this device.

        Args:
            title (str): title of device
        """
        if self.robot is None or self.robot.world is None:
            raise Exception("can't watch device until added to robot, and robot is in world")

        # Make the watcher if doesn't exist:
        self.get_widget(**kwargs)
        self._watcher.watch()

    def get_widget(self, **kwargs):
        """
        Return the dynamically updating widget.

        Args:
            title (str): title of device
        """
        from ..watchers import CameraWatcher

        if self._watcher is None:
            self._watcher = CameraWatcher(self, **kwargs)
            self.robot.world._watchers.append(self._watcher)
            return self._watcher.get_widget()
        else:
            return self._watcher.get_widget(**kwargs)

    def _step(self, time_step):
        pass

    def _get_visible_area(self):
        """
        What are the ranges of the field of view?
        Return a list of (p1, p2) that represent lines
        across the background, from front to back,
        down the edges of the field of view.
        """
        step = 1
        all_points = []
        for angle in [self.fov / 2, -self.fov / 2]:
            points = []
            dx, dy = rotate_around(0, 0, step, self.robot.a + angle)
            cx, cy = self.robot.x, self.robot.y
            x, y = cx, cy
            for i in range(0, self.max_range, step):
                points.append((x, y))
                x += dx
                y += dy
            all_points.append(points)
        return zip(*all_points)

    def update(self, draw_list=None):
        """
        Cameras operate in a lazy way: they don't actually update
        until needed because they are so expensive.

        Args:
            draw_list (list): optional. If given, then the
                method can add to it for drawing later.
        """
        pass

    def _update(self):
        # Update timestamp:
        self.time = self.robot.world.time

        for i in range(self.cameraShape[0]):
            angle = i / self.cameraShape[0] * self.fov - self.fov / 2
            self.hits[i] = self.robot.cast_ray(
                self.robot.x,
                self.robot.y,
                PI_OVER_2 - self.robot.a - angle,
                self.max_range,
            )


    def draw(self, backend):
        """
        Draw the device on the backend.

        Args:
            backend (Backend): an aitk drawing backend

        NOTE: Currently, cameras are fixed at 0,0 and face forwards.
        """
        backend.set_fill(Color(0, 64, 0))
        backend.strokeStyle(None, 0)
        backend.draw_rect(5.0, -3.33, 1.33, 6.33)

        # Note angle in sim world is opposite in graphics:
        hits = self.robot.cast_ray(
            self.robot.x,
            self.robot.y,
            PI_OVER_2 - self.robot.a + self.fov / 2,
            self.max_range,
        )
        if hits:
            p = rotate_around(0, 0, hits[-1].distance, -self.fov / 2,)
        else:
            p = rotate_around(0, 0, self.max_range, -self.fov / 2,)
        backend.draw_line(0, 0, p[0], p[1])

        # Note angle in sim world is opposite in graphics:
        hits = self.robot.cast_ray(
            self.robot.x,
            self.robot.y,
            PI_OVER_2 - self.robot.a - self.fov / 2,
            self.max_range,
        )
        if hits:
            p = rotate_around(0, 0, hits[-1].distance, self.fov / 2,)
        else:
            p = rotate_around(0, 0, self.max_range, self.fov / 2,)
        backend.draw_line(0, 0, p[0], p[1])

    def _find_closest_wall(self, hits):
        for hit in reversed(hits):  # reverse make it closest first
            if hit.height < 1.0:  # skip non-walls
                continue
            return hit.distance
        return float("inf")

    def _get_ground_color(self, area, i, j):
        if self.robot.world._ground_image is not None and area is not None:
            # i is width ray (camera width),
            # j is distance (height of camera/2, 64 to 128)
            dist = round(
                ((self.cameraShape[1] - j) / self.cameraShape[1] / 3) * len(area)
            )
            visible_width_points = area[dist]
            p1, p2 = visible_width_points
            # get a position i/width on line
            # minx, maxx = sorted([p1[0], p2[0]])
            # miny, maxy = sorted([p1[1], p2[1]])
            spanx = abs(p1[0] - p2[0])
            spany = abs(p1[1] - p2[1])
            if p1[0] < p2[0]:
                x = round(
                    (p1[0] + spanx * (1.0 - i / self.cameraShape[0]))
                    * self.robot.world.scale
                )
            else:
                x = round(
                    (p1[0] - spanx * (1.0 - i / self.cameraShape[0]))
                    * self.robot.world.scale
                )
            if p1[1] < p2[1]:
                y = round(
                    (p1[1] + spany * (1.0 - i / self.cameraShape[0]))
                    * self.robot.world.scale
                )
            else:
                y = round(
                    (p1[1] - spany * (1.0 - i / self.cameraShape[0]))
                    * self.robot.world.scale
                )
            # find that pixel
            if (0 <= x < (self.robot.world.width - 1) * self.robot.world.scale) and (
                0 <= y < (self.robot.world.height - 1) * self.robot.world.scale
            ):
                # c = Color(*self.robot.world._ground_image_pixels[(x, y)])
                # self.robot.world._ground_image_pixels[(x, y)] = (0, 0, 0)
                # return c

                sum = Color(0)
                count = 0
                # Need more sampling as distance increases:
                for j in range(self.samples):
                    try:
                        c = Color(*self.robot.world._ground_image_pixels[(x + j, y)])
                        # self.robot.world._ground_image_pixels[(x + j, y)] = (0, 0, 0)
                        count += 1
                    except Exception:
                        continue
                    sum += c
                return sum / count

        return self.robot.world.ground_color

    def display(self, type="color"):
        """
        Display an image of the camera.

        Args:
            type (str): the return type of image. Can be "color"
                or "depth"
        """
        image = self.get_image(type=type)
        display(image)

    def get_image(self, type="color"):
        """
        Get an image of the camera.

        Args:
            type (str): the return type of image. Can be "color"
                or "depth"
        """
        try:
            from PIL import Image
        except ImportError:
            print("Pillow (PIL) module not available; get_image() unavailable")
            return

        # Lazy; only get the data when we need it:
        self._update()
        if self.robot.world._ground_image is not None:
            area = list(self._get_visible_area())
        else:
            area = None
        pic = Image.new("RGBA", (self.cameraShape[0], self.cameraShape[1]))
        pic_pixels = pic.load()
        # FIXME: probably should have a specific size rather than scale it to world
        size = max(self.robot.world.width, self.robot.world.height)
        hcolor = None
        # draw non-robot walls first:
        for i in range(self.cameraShape[0]):
            hits = [hit for hit in self.hits[i] if hit.height == 1.0]  # only walls
            if len(hits) == 0:
                continue
            hit = hits[-1]  # get closest
            high = None
            hcolor = None
            if hit:
                if self.fov < PI_OVER_2:
                    # Orthoginal distance to camera:
                    angle = hit.angle
                    hit_distance = abs(hit.distance * math.sin(angle))
                else:
                    hit_distance = hit.distance

                distance_ratio = 1.0 - hit_distance / size
                s = distance_ratio * self.sizeFadeWithDistance
                sc = distance_ratio * self.colorsFadeWithDistance
                if type == "color":
                    r = hit.color.red * sc
                    g = hit.color.green * sc
                    b = hit.color.blue * sc
                elif type == "depth":
                    r = 255 * distance_ratio
                    g = 255 * distance_ratio
                    b = 255 * distance_ratio
                else:
                    avg = (hit.color.red + hit.color.green + hit.color.blue) / 3.0
                    r = avg * sc
                    g = avg * sc
                    b = avg * sc
                hcolor = Color(r, g, b)
                high = (1.0 - s) * self.cameraShape[1]
            else:
                high = 0

            horizon = self.cameraShape[1] / 2
            for j in range(self.cameraShape[1]):
                dist = max(min(abs(j - horizon) / horizon, 1.0), 0.0)
                if j < high / 2:  # sky
                    if type == "depth":
                        if self.reflectSky:
                            color = Color(255 * dist)
                        else:
                            color = Color(0)
                    elif type == "color":
                        color = Color(0, 0, 128)
                    else:
                        color = Color(128 / 3)
                    pic_pixels[i, j] = color.to_tuple()
                elif j < self.cameraShape[1] - high / 2:  # hit
                    if hcolor is not None:
                        pic_pixels[i, j] = hcolor.to_tuple()
                else:  # ground
                    if type == "depth":
                        if self.reflectGround:
                            color = Color(255 * dist)
                        else:
                            color = Color(0)
                    elif type == "color":
                        color = self._get_ground_color(area, i, j)
                    else:
                        color = Color(128 / 3)
                    pic_pixels[i, j] = color.to_tuple()

        # Other robots, draw on top of walls:
        self.obstacles = {}
        for i in range(self.cameraShape[0]):
            closest_wall_dist = self._find_closest_wall(self.hits[i])
            hits = [hit for hit in self.hits[i] if hit.height < 1.0]  # obstacles
            for hit in hits:
                if hit.distance > closest_wall_dist:
                    # Behind wall, try next
                    continue

                if self.fov < PI_OVER_2:
                    angle = hit.angle
                    hit_distance = abs(hit.distance * math.sin(angle))
                else: # perspective
                    hit_distance = hit.distance

                distance_ratio = 4.0 - hit_distance / size
                s = distance_ratio * self.sizeFadeWithDistance
                sc = distance_ratio * self.colorsFadeWithDistance
                distance_to = self.cameraShape[1] / 2 * (4.0 - sc)
                # scribbler was 30, so 0.23 height ratio
                # height is ratio, 0 to 1
                height = round(hit.height * self.cameraShape[1] / 2.0 * s)
                if type == "color":
                    r = hit.color.red * sc
                    g = hit.color.green * sc
                    b = hit.color.blue * sc
                elif type == "depth":
                    r = 255 * distance_ratio
                    g = 255 * distance_ratio
                    b = 255 * distance_ratio
                else:
                    avg = (hit.color.red + hit.color.green + hit.color.blue) / 3.0
                    r = avg * sc
                    g = avg * sc
                    b = avg * sc
                hcolor = Color(r, g, b)
                horizon = self.cameraShape[1] / 2
                self._record_obstacle(
                    hit.robot,
                    i,
                    self.cameraShape[1] - 1 - round(distance_to),
                    self.cameraShape[1] - height - 1 - 1 - round(distance_to),
                )
                if not hit.robot.has_image():
                    for j in range(height):
                        pic_pixels[
                            i, self.cameraShape[1] - j - 1 - round(distance_to)
                        ] = hcolor.to_tuple()
        if self.bulbs_are_visible:
            self._show_bulbs(pic)
        if self.food_is_visible:
            self._show_food(pic)
        self._show_obstacles(pic)
        return pic

    def _show_bulbs(self, image):
        from PIL import ImageDraw, Image

        draw_list = []
        for bulb in self.robot.world._get_light_sources(all=True):
            if bulb.robot is self.robot:
                # You can't sense your own bulbs
                continue

            # cast a ray, see if it hits this robot
            # if so, we can see it
            # ignore boundary boxes around robots that contain the bulb
            bulb_x, bulb_y = bulb.get_position()
            hits = self.robot.cast_ray(bulb_x, bulb_y, 0, self.max_range,
                                       self.robot.x, self.robot.y, ignore_robots=[bulb.robot,
                                                                                  self.robot])
            if len(hits) == 0:
                draw_list.append((bulb, bulb_x, bulb_y))

        if len(draw_list) > 0:
            layer = Image.new('RGBA', self.cameraShape, (0, 0, 0, 0))
            drawing = ImageDraw.Draw(layer, "RGBA")
            size = max(self.robot.world.width, self.robot.world.height)
            color = Color(bulb.color)
            #color.alpha = 128
            for (bulb, bulb_x, bulb_y) in draw_list:
                angle = uniform_angle(math.pi * 3/2 - math.atan2(self.robot.x - bulb_x,
                                                                 self.robot.y - bulb_y))
                min_angle = uniform_angle(self.robot.a - self.fov/2)
                max_angle = uniform_angle(self.robot.a + self.fov/2)
                if max_angle < min_angle:
                    if 0 <= angle < max_angle:
                        angle += TWO_PI
                    max_angle += TWO_PI

                if min_angle < angle < max_angle:
                    span = max_angle - min_angle
                    x = int((angle - min_angle)/span * self.cameraShape[0])
                    hit_distance = distance(self.robot.x, self.robot.y, bulb_x, bulb_y)
                    distance_ratio = 1.0 - hit_distance / size
                    s = distance_ratio * self.sizeFadeWithDistance
                    high = (1.0 - s) * self.cameraShape[1]
                    y = (self.cameraShape[1] / 2) + high / 2
                    radius = 5
                    drawing.ellipse((x - radius, y - radius,
                                     x + radius, y + radius),
                                    fill=color.to_tuple())
            image.paste(Image.alpha_composite(image, layer))

    def _show_food(self, image):
        from PIL import ImageDraw, Image

        draw_list = []
        for food in self.robot.world._food:
            hits = self.robot.cast_ray(food.x, food.y, 0, self.max_range,
                                       self.robot.x, self.robot.y)
            if len(hits) == 0:
                draw_list.append((food.x, food.y, food.standard_deviation))

        if len(draw_list) > 0:
            layer = Image.new('RGBA', self.cameraShape, (0, 0, 0, 0))
            drawing = ImageDraw.Draw(layer, "RGBA")
            size = max(self.robot.world.width, self.robot.world.height)
            color = Color(WHITE)
            #color.alpha = 128
            for (x, y, sd) in draw_list:
                angle = uniform_angle(math.pi * 3/2 - math.atan2(self.robot.x - x,
                                                                 self.robot.y - y))
                min_angle = uniform_angle(self.robot.a - self.fov/2)
                max_angle = uniform_angle(self.robot.a + self.fov/2)
                if max_angle < min_angle:
                    if 0 <= angle < max_angle:
                        angle += TWO_PI
                    max_angle += TWO_PI

                if min_angle < angle < max_angle:
                    span = max_angle - min_angle
                    x = int((angle - min_angle)/span * self.cameraShape[0])
                    hit_distance = distance(self.robot.x, self.robot.y, x, y)
                    distance_ratio = 1.0 - hit_distance / size
                    s = distance_ratio * self.sizeFadeWithDistance
                    high = (1.0 - s) * self.cameraShape[1]
                    y = (self.cameraShape[1] / 2) + high / 2
                    radius = 5
                    drawing.rectangle((x - radius, y - radius,
                                       x + radius, y + radius),
                                      fill=color.to_tuple())
            image.paste(Image.alpha_composite(image, layer))

    def _show_obstacles(self, image):
        # FIXME: show back to front
        # FIXME: how to show when partially behind wall?
        # data = {"robot": obj.has_image(), "min_x", "min_y", "max_x", "max_y"}
        for data in self.obstacles.values():
            if data["robot"].has_image():
                # the angle to me + offset for graphics + the robot angle:
                radians = (
                    math.atan2(
                        data["robot"].x - self.robot.x, data["robot"].y - self.robot.y
                    )
                    + PI_OVER_2
                    + data["robot"].a
                )
                degrees = round(radians * ONE80_OVER_PI)
                picture = data["robot"].get_image_3d(degrees)  # degrees
                x1, y1 = data["min_x"], data["min_y"]  # noqa: F841
                x2, y2 = data["max_x"], data["max_y"]
                try:  # like too small
                    picture.thumbnail((x2 - x1, 10000))  # to keep aspect ratio
                    # picture = picture.resize((x2 - x1, y2 - y1))
                    x3 = x2 - picture.height
                    y3 = y2 - picture.width
                    image.paste(picture, (x3, y3), picture)
                except Exception:
                    print("Exception in processing image")

    def _record_obstacle(self, robot, x, y1, y2):
        if robot.name not in self.obstacles:
            self.obstacles[robot.name] = {
                "robot": robot,
                "max_x": float("-inf"),
                "max_y": float("-inf"),
                "min_x": float("inf"),
                "min_y": float("inf"),
            }
        self.obstacles[robot.name]["max_x"] = max(
            self.obstacles[robot.name]["max_x"], x
        )
        self.obstacles[robot.name]["min_x"] = min(
            self.obstacles[robot.name]["min_x"], x
        )
        self.obstacles[robot.name]["max_y"] = max(
            self.obstacles[robot.name]["max_y"], y1, y2
        )
        self.obstacles[robot.name]["min_y"] = min(
            self.obstacles[robot.name]["min_y"], y1, y2
        )

    def get_point_cloud(self):
        """
        Get a 3D point cloud from the camera data.

        Returns a list of [x, y, distance, red, green, blue]
        for each (x,y) of the camera.
        """
        depth_pic = self.get_image("depth")
        depth_pixels = depth_pic.load()
        color_pic = self.get_image("color")
        color_pixels = color_pic.load()
        points = []
        for x in range(self.cameraShape[0]):
            for y in range(self.cameraShape[1]):
                dist_color = depth_pixels[x, y]
                color = color_pixels[x, y]
                if dist_color[0] != 255:
                    points.append(
                        [
                            self.cameraShape[0] - x - 1,
                            self.cameraShape[1] - y - 1,
                            dist_color[0],
                            color[0],
                            color[1],
                            color[2],
                        ]
                    )
        return points

    def get_fov(self):
        """
        Get the field of view angle in degrees.
        """
        return self.fov * ONE80_OVER_PI

    def set_fov(self, angle):
        """
        Set the field of view angle in degrees of the camera.

        Args:
            angle (float): angle in degrees of field of view
        """
        # given in degrees
        # save in radians
        # scale = min(max(angle / 6.0, 0.0), 1.0)
        self.fov = angle * PI_OVER_180
        # self.sizeFadeWithDistance = scale
        self.reset()

    def set_size(self, width, height):
        """
        Set the height and width of the camera in pixels.

        Args:
            width (int): width of camera in pixels
            height (int): height of camera in pixels
        """
        self.cameraShape[0] = width
        self.cameraShape[1] = height
        self.reset()

    def set_max(self, max_range):
        """
        Set the maximum distance the camera can see.

        Args:
            max_range (float): distance (in CM) the camera can see
        """
        self.max_range = max_range

    def set_width(self, width):
        """
        Set the width of the camera in pixels.

        Args:
            width (int): width of camera in pixels
        """
        self.cameraShape[0] = width
        self.reset()

    def set_height(self, height):
        """
        Set the height of the camera in pixels.

        Args:
            height (int): height of camera in pixels
        """
        self.cameraShape[1] = height
        self.reset()

    def set_name(self, name):
        """
        Set the name of the camera.

        Args:
            name (str): the name of the camera
        """
        self.name = name

    def get_name(self):
        """
        Get the name of the camera.
        """
        return self.name

    def get_width(self):
        """
        Get the width in pixels of the camera.
        """
        return self.cameraShape[0]

    def get_height(self):
        """
        Get the height in pixels of the camera.
        """
        return self.cameraShape[1]

    def get_max(self):
        """
        Get the maximum distance in CM the camera can see.
        """
        return self.max_range


class GroundCamera(Camera):
    def __init__(self, width=15, height=15, name="ground-camera", **kwargs):
        """
        A downward-facing camera device.

        Args:
            width (int): width of camera in pixels
            height (int): height of camera in pixels
            name (str): the name of the camera
        """
        config = {
            "width": width,
            "height": height,
            "name": name,
        }
        self._watcher = None
        self.robot = None
        self.initialize()
        self.from_json(config)

    def initialize(self):
        """
        Internal method to set all settings to default values.
        """
        self.type = "ground-camera"
        self.time = 0.0
        self.cameraShape = [15, 15]
        self.name = "ground-camera"

    def from_json(self, config):
        """
        Set the settings from a device config.

        Args:
            config (dict): a config dictionary
        """
        if "width" in config:
            self.cameraShape[0] = config["width"]
        if "height" in config:
            self.cameraShape[1] = config["height"]
        if "name" in config:
            self.name = config["name"]

    def to_json(self):
        """
        Save the internal settings to a config dictionary.
        """
        return {
            "class": self.__class__.__name__,
            "width": self.cameraShape[0],
            "height": self.cameraShape[1],
            "name": self.name,
        }

    def __repr__(self):
        return "<GroundCamera %r size=(%r,%r)>" % (
            self.name,
            self.cameraShape[0],
            self.cameraShape[1],
        )

    def update(self, draw_list=None):
        """
        Update the device.

        Args:
            draw_list (list): optional. If given, then the
                method can add to it for drawing later.

        Cameras operate in a lazy way: they don't actually update
        until needed because they are so expensive.
        """
        pass

    def draw(self, backend):
        """
        Draw the device on the backend.

        Args:
            backend (Backend): an aitk drawing backend
        """
        left = -(self.cameraShape[0] // 2) / self.robot.world.scale
        upper = -(self.cameraShape[1] // 2) / self.robot.world.scale
        right = (self.cameraShape[0] // 2) / self.robot.world.scale
        lower = (self.cameraShape[1] // 2) / self.robot.world.scale
        backend.strokeStyle(Color(128), 1)
        backend.draw_line(left, upper, right, upper)
        backend.draw_line(right, upper, right, lower)
        backend.draw_line(right, lower, left, lower)
        backend.draw_line(left, lower, left, upper)

    def get_image(self, type=None):
        """
        Get an image of the camera.

        Args:
            type (str): the return type of image. Can be "color"
                or "depth"
        """
        # FIXME: would be faster to trim image down
        # before rotating
        center = (
            self.robot.x * self.robot.world.scale,
            self.robot.y * self.robot.world.scale,
        )
        rotated_image = self.robot.world._ground_image.rotate(
            (self.robot.a - math.pi / 4 * 6) * (ONE80_OVER_PI), center=center,
        )
        left = center[0] - self.cameraShape[0] // 2
        right = center[0] + self.cameraShape[0] // 2
        upper = center[1] - self.cameraShape[1] // 2
        lower = center[1] + self.cameraShape[1] // 2
        return rotated_image.crop((left, upper, right, lower))
