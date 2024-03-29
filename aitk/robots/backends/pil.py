# -*- coding: utf-8 -*-
# ************************************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
# ************************************************************

import io
import math

from PIL import Image, ImageDraw, ImageFont

from aitk.utils import get_font, Color
from ..utils import arange, distance
from .base import Backend

BLACK = Color("black")
WHITE = Color("white")

DEFAULT_FONT_NAMES = (
    get_font("FreeMonoBold.ttf"),
    "arial.ttf",
    "Arial.ttf",
    "NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/Library/Fonts/Arial.ttf",
)


class PILBackend(Backend):
    # Specific to this class:

    def __init__(self, *args, **kwargs):
        self.widget = None
        super().__init__(*args, **kwargs)

    def initialize(self, **kwargs):
        self.matrix = []
        self.kwargs = kwargs
        self.font = None
        self.font_size = kwargs.get("font_size", int(12 * self._scale))
        for font_string_name in DEFAULT_FONT_NAMES:
            try:
                self.font = ImageFont.truetype(font_string_name, self.font_size)
                break
            except OSError:
                continue

        self.mode = kwargs.get("mode", "RGBA")  # "RGBA" or "RGB"
        self.format = kwargs.get(
            "format", "jpeg"
        )  # "png", "gif", or "jpeg" # png,gif doesn't do opacity?!

        if self.mode == "RGBA" and self.format == "jpeg":
            self.mode = "RGB"
            kwargs["mode"] = "RGB"

        self.status_height = 14
        self.image = Image.new(
            self.mode,
            size=(int(self.width * self._scale), int((self.height  + self.status_height) * self._scale)),
        )
        self.draw = ImageDraw.Draw(self.image, "RGBA")
        if self.font:
            text_width, text_height = self.draw.textsize("0", self.font)
            self.char_width = text_width / self._scale
            self.char_height = text_height / self._scale
        else:
            self.char_width = 5.8
            self.char_height = 10

    def update_dimensions(self, width, height, scale):
        if width != self.width or height != self.height or self._scale != scale:
            self.width = width
            self.height = height
            self._scale = scale
            self.initialize(**self.kwargs)

    # Canvas API:

    def draw_status(self, text):
        self.set_fill(BLACK)
        self.draw_rect(
            0,
            self.height,
            self.width,
            self.status_height,
        )
        self.set_fill(WHITE)
        self.text(text, 1, self.height - 1)

    def to_image(self, format=None):
        format = format if format is not None else self.format
        fp = io.BytesIO()
        self.image.save(fp, format=format)
        return fp.getvalue()

    def get_widget(self, width=None, height=None):
        from ipywidgets import Image

        if self.widget is None:
            self.widget = Image(value=self.to_image())
            self.widget.layout.margin = "auto"
            self.widget.layout.border = "10px solid rgb(0 177 255)"

        if width is not None:
            self.widget.layout.width = width
        if height is not None:
            self.widget.layout.height = height

        return self.widget

    def draw_watcher(self):
        if self.widget:
            self.widget.value = self.to_image()

    def flush(self):
        pass

    def get_image(self, time):
        return self.image

    # High-level API:

    def draw_image(self, image, x, y):
        self.image.paste(image, (x, y))

    def get_color(self, color):
        if isinstance(color, Color):
            return color.to_tuple()
        elif color != "":
            return color
        else:
            return None

    def get_line_width(self):
        return round(self.line_width * self._scale)

    def get_style(self, style):
        if style == "fill":
            return self.get_color(self.fill_style)
        elif style == "stroke":
            return self.get_color(self.stroke_style)

    def p(self, x, y):
        # Transform a point
        for matrix in self.matrix:
            for transform in reversed(matrix):
                if transform[0] == "translate":
                    x += transform[1]
                    y += transform[2]
                elif transform[0] == "rotate":
                    dist = distance(0, 0, x, y)
                    angle2 = math.atan2(-x, y)
                    angle = transform[1]
                    x = dist * math.cos(angle2 + angle + math.pi / 2)
                    y = dist * math.sin(angle2 + angle + math.pi / 2)
        return x * self._scale, y * self._scale

    def r(self, angle):
        # Transform an angle
        for matrix in self.matrix:
            for transform in reversed(matrix):
                if transform[0] == "translate":
                    pass
                elif transform[0] == "rotate":
                    angle += transform[1]
        return angle

    def draw_lines(self, points, stroke_style=None):
        self.stroke_style = stroke_style
        for i in range(len(points)):
            if i < len(points) - 2:
                self.draw_line(
                    points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]
                )

    def draw_line(self, x1, y1, x2, y2):
        p1x, p1y = self.p(x1, y1)
        p2x, p2y = self.p(x2, y2)
        self.draw.line(
            (p1x, p1y, p2x, p2y),
            fill=self.get_style("stroke"),
            width=self.get_line_width(),
        )

    def clear(self):
        # self.fill_style = "white"
        self.draw_rect(0, 0, self.width, self.height)

    def text(self, t, x, y):
        x, y = self.p(x, y)
        self.draw.text((x, y), t, fill=self.get_style("fill"), font=self.font)

    def pushMatrix(self):
        self.matrix.append([])

    def popMatrix(self):
        self.matrix.pop()

    def scale(self, x, y):
        pass

    def resetScale(self):
        pass

    def draw_rect(self, x, y, width, height):
        p1x, p1y = self.p(x, y)
        p2x, p2y = self.p(x + width, y)
        p3x, p3y = self.p(x + width, y + height)
        p4x, p4y = self.p(x, y + height)

        self.draw.polygon(
            (p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y),
            fill=self.get_style("fill"),
            outline=self.get_style("stroke"),
        )

    def draw_ellipse(self, x, y, radiusX, radiusY):
        # Given as center and radius
        if radiusX == radiusY:
            x, y = self.p(x, y)

            p1x, p1y = (x - radiusX * self._scale, y - radiusY * self._scale)
            p2x, p2y = (x + radiusX * self._scale, y + radiusY * self._scale)

            minx = min(p1x, p2x)
            miny = min(p1y, p2y)
            maxx = max(p1x, p2x)
            maxy = max(p1y, p2y)

            self.draw.ellipse(
                (minx, miny, maxx, maxy),
                fill=self.get_style("fill"),
                outline=self.get_style("stroke"),
                width=self.get_line_width(),
            )
        else:
            self.draw_arc(x, y, radiusX, radiusY, 0, math.pi * 2, 12)

    def draw_arc(self, x, y, width, height, startAngle, endAngle, segments=5):
        points = [self.p(x, y)]

        for angle in arange(startAngle, endAngle, (endAngle - startAngle) / segments):
            point = (
                x + height * math.cos(angle),
                y + width * math.sin(angle),
            )
            points.append(self.p(point[0], point[1]))

        self.draw.polygon(
            points, fill=self.get_style("fill"),
        )

        self.draw.line(
            points[1:], fill=self.get_style("stroke"), width=self.get_line_width()
        )

    def beginShape(self):
        self.points = []

    def endShape(self):
        self.draw.polygon(
            self.points, fill=self.get_style("fill"), outline=self.get_style("stroke")
        )

    def vertex(self, x, y):
        self.points.append(self.p(x, y))

    def translate(self, x, y):
        self.matrix[-1].append(("translate", x, y))

    def rotate(self, angle):
        self.matrix[-1].append(("rotate", angle))
