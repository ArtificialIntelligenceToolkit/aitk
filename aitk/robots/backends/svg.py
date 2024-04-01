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

from PIL import Image
from svgwrite import Drawing

from .base import Backend

from aitk.utils import Color


# First attempt, replicating the low-level SVG badly
# Second attempt should implement the HIGH-LEVEL API


class SVGBackend(Backend):
    # Specific to this class:

    def initialize(self):
        self.widget = None
        self.char_width = 6.8
        self.char_height = 11
        self.reset()

    def reset(self):
        self.stack = []
        self.points = []
        dwg = Drawing("canvas.svg", (self.width, self.height))
        dwg.viewbox(0, 0, self.width, self.height)
        self.stack.append(dwg)

    # Overrides:

    def watch(self):
        from ipywidgets import HTML

        if self.widget is None:
            self.widget = HTML(value=self.stack[0]._repr_svg_(),)

        return self.widget

    def draw_watcher(self):
        if self.widget:
            self.widget.value = self.stack[0]._repr_svg_()

    def update_dimensions(self, width, height, scale):
        # No need, SVG handles this
        pass

    def flush(self):
        pass

    def get_image(self, time):
        try:
            import cairosvg
        except ImportError:
            print("This backend.get_image() requires cairosvg")
            return

        bytes = cairosvg.svg2png(self.stack[0].tostring(), scale=self._scale)
        fp = io.BytesIO(bytes)
        picture = Image.open(fp)
        if picture.mode == "RGB":
            picture = picture.convert("RGBA")
        return picture

    # Low-level API:

    def set_stroke_style(self, color):
        self.stroke_style = color
        self.stroke_style_color = Color(0, 0, 0, 255)

    def set_fill_style(self, color):
        self.fill_style = color.rgb()
        self.fill_style_color = color

    def fill_opacity(self):
        alpha = self.fill_style_color.alpha
        if alpha != 255:
            return round(alpha / 255, 2)

    def stroke_opacity(self):
        alpha = self.stroke_style_color.alpha
        if alpha != 255:
            return round(alpha / 255, 2)

    def get_style(self, *items):
        fo = self.fill_opacity()
        so = self.stroke_opacity()
        map = {
            "fill": ("fill:%s" % self.fill_style) if self.fill_style else None,
            "stroke-width": "stroke-width:%s" % self.line_width,
            "stroke": ("stroke:%s" % self.stroke_style) if self.stroke_style else None,
            "fill-opacity": ("fill-opacity:%s" % fo) if fo is not None else None,
            "stroke-opacity": ("stroke-opacity:%s" % so) if so is not None else None,
        }
        styles = []
        for item in items:
            v = map[item]
            if v:
                styles.append(v)
        style = ";".join(styles)
        return style

    def arc(self, x, y, radius, startAngle, endAngle):
        def polarToCartesian(centerX, centerY, radius, angleInRadians):
            return [
                round(centerX + (radius * math.cos(angleInRadians)), 2),
                round(centerY + (radius * math.sin(angleInRadians)), 2),
            ]

        start = polarToCartesian(x, y, radius, endAngle)
        end = polarToCartesian(x, y, radius, startAngle)

        if len(self.points) > 0:
            path = [
                "M",
                start[0],
                start[1],
                "A",
                round(radius, 2),
                round(radius, 2),
                0,
                0,
                0,
                end[0],
                end[1],
                "L",
                self.points[-1][0],
                self.points[-1][1],
            ]
            styles = ["fill", "fill-opacity"]
        else:
            path = [
                "M",
                start[0],
                start[1],
                "A",
                round(radius, 2),
                round(radius, 2),
                0,
                0,
                0,
                end[0],
                end[1],
            ]
            styles = ["stroke", "stroke-width"]
        path = [str(item) for item in path]
        d = " ".join(path)
        dwg = self.stack[-1]
        style = self.get_style(*styles)
        dwg.add(self.stack[0].path(d=d, style=style))

    def clear_rect(self, x, y, width, height):
        self.reset()

    def text(self, text, x, y):
        self.fill_text(text, x, y + self.char_height)

    def fill_text(self, text, x, y):
        style = self.get_style("fill")
        dwg = self.stack[-1]
        dwg.add(self.stack[0].text(text=text, insert=(x, y), style=style))

    def fill_rect(self, x, y, width, height):
        style = self.get_style("fill")
        self.stack[-1].add(
            self.stack[0].rect(
                insert=(round(x, 2), round(y, 2)),
                size=(round(width, 2), round(height, 2)),
                style=style,
            )
        )

    def fill(self):
        if len(self.points) >= 2:
            style = self.get_style("fill")
            dwg = self.stack[-1]
            dwg.add(self.stack[0].polygon(points=self.points, style=style))
            self.points.clear()

    def stroke(self):
        if len(self.points) >= 2:
            style = self.get_style("stroke", "stroke-width", "stroke-opacity")
            dwg = self.stack[-1]
            last = self.points[0]
            for point in self.points[1:]:
                dwg.add(self.stack[0].line(start=last, end=point, style=style))
                last = point
            self.points.clear()

    def move_to(self, x, y):
        self.points = [(round(x, 2), round(y, 2))]

    def line_to(self, x, y):
        self.points.append([round(x, 2), round(y, 2)])

    def save(self):
        dwg = self.stack[0]
        self.stack.append(dwg.add(dwg.g()))

    def restore(self):
        parent = self.stack[-2]
        dwg = self.stack[-1]
        parent.add(dwg)
        self.stack.pop()

    def translate(self, x, y):
        self.stack[-1].translate(x, y)

    def scale(self, xscale, yscale):
        pass

    def set_transform(self, x, y, z, a, b, c):
        pass

    def rotate(self, angle):
        # comes in radians, need to convert to degrees
        self.stack[-1].rotate(angle * 180 / math.pi)

    def begin_path(self):
        pass

    def close_path(self):
        pass

    def ellipse(self, x, y, radiusX, radiusY, a, b, angle):
        dwg = self.stack[-1]
        style = self.get_style("fill")
        dwg.add(
            self.stack[0].ellipse(
                center=(round(x, 2), round(y, 2)),
                r=(round(radiusX, 2), round(radiusY, 2)),
                style=style,
            )
        )

    def get_image_data(self):
        pass

    def put_image_data(self, scaled, x, y):
        pass

    def create_image_data(sefl, width, height):
        pass
