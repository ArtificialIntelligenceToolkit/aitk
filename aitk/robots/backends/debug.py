# -*- coding: utf-8 -*-
# ************************************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
# ************************************************************

from .base import Backend

# Implements all of the needed methods:


class DebugBackend(Backend):

    # Canvas API:
    def initialize(self, **kwargs):
        self.show_high = kwargs.get("show_high", True)
        self.show_low = kwargs.get("show_low", True)

    def flush(self):
        print("flush()")

    def get_image(self, time):
        print("get_image()")

    def watch(self):
        print("watch()")

    # High-level API (backend draw API)

    def draw_image(self, image, x, y):
        if self.show_high:
            print("draw_image%r" % ((image, x, y)))
        super().draw_image(image, x, y)

    def draw_lines(self, points, stroke_style=None):
        if self.show_high:
            print("draw_lines%r" % ((points, stroke_style),))
        super().draw_lines(points, stroke_style)

    def set_stroke_style(self, color):
        if self.show_high:
            print("set_stroke_style%r" % ((color),))
        super().set_stroke_style(color)

    def set_fill_style(self, color):
        if self.show_high:
            print("set_fill_style%r" % ((color),))
        super().set_fill_style(color)

    def clear(self):
        if self.show_high:
            print("clear()")
        super().clear()

    def set_font(self, style):
        if self.show_high:
            print("set_font%r" % ((style,),))
        super().set_font(style)

    def text(self, t, x, y):
        if self.show_high:
            print("text%r" % ((t, x, y),))
        super().text(t, x, y)

    def lineWidth(self, width):
        if self.show_high:
            print("lineWidth%r" % ((width),))
        super().lineWidth(width)

    def strokeStyle(self, color, width):
        if self.show_high:
            print("strokeStyle%r" % ((color, width),))
        super().strokeStyle(color, width)

    def make_stroke(self):
        if self.show_high:
            print("make_stroke()")
        super().make_stroke()

    def noStroke(self):
        if self.show_high:
            print("noStroke()")
        super().noStroke()

    def set_fill(self, color):
        if self.show_high:
            print("set_fill%r" % ((color),))
        super().set_fill(color)

    def noFill(self):
        if self.show_high:
            print("noFill()")
        super().noFill()

    def draw_line(self, x1, y1, x2, y2):
        if self.show_high:
            print("draw_line%r" % ((x1, y1, x2, y2),))
        super().draw_line(x1, y1, x2, y2)

    def pushMatrix(self):
        if self.show_high:
            print("pushMatrix()")
        super().pushMatrix()

    def popMatrix(self):
        if self.show_high:
            print("popMatrix()")
        super().popMatrix()

    def resetScale(self):
        if self.show_high:
            print("resetScale()")
        super().resetScale()

    def beginShape(self):
        if self.show_high:
            print("beginShape()")
        super().beginShape()

    def endShape(self):
        if self.show_high:
            print("endShape()")
        super().endShape()

    def vertex(self, x, y):
        if self.show_high:
            print("vertex%r" % ((x, y),))
        super().vertex(x, y)

    def draw_rect(self, x, y, width, height):
        if self.show_high:
            print("draw_rect%r" % ((x, y, width, height),))
        super().draw_rect(x, y, width, height)

    def draw_circle(self, x, y, radius):
        if self.show_high:
            print("draw_cirle%r" % ((x, y, radius),))
        super().draw_circle(x, y, radius)

    def draw_ellipse(self, x, y, radiusX, radiusY):
        if self.show_high:
            print("draw_ellipse%r" % ((x, y, radiusX, radiusY),))
        super().draw_ellipse(x, y, radiusX, radiusY)

    def draw_arc(self, x, y, width, height, startAngle, endAngle):
        if self.show_high:
            print("draw_arc%r" % ((x, y, width, height, startAngle, endAngle),))
        super().draw_arc(x, y, width, height, startAngle, endAngle)

    # Low-level API (HTML Canvas API):

    def arc(self, x, y, width, startAngle, endAngle):
        if self.show_low:
            print("    arc(%r,%r,%r,%r,%r)" % (x, y, width, startAngle, endAngle))

    def get_image_data(self):
        if self.show_low:
            print("    get_image_data()")

    def clear_rect(self, x, y, width, height):
        if self.show_low:
            print("    clear_rect(", x, y, width, height, ")")

    def fill_text(self, t, x, y):
        if self.show_low:
            print("    fill_text(", t, x, y, ")")

    def fill_rect(self, x, y, width, height):
        if self.show_low:
            print("    fill_rect(", x, y, width, height, ")")

    def fill(self):
        if self.show_low:
            print("    fill()")

    def stroke(self):
        if self.show_low:
            print("    stroke()")

    def move_to(self, x, y):
        if self.show_low:
            print("    move_to(", x, y, ")")

    def line_to(self, x, y):
        if self.show_low:
            print("    line_to(", x, y, ")")

    def save(self):
        if self.show_low:
            print("    save()")

    def restore(self):
        if self.show_low:
            print("    restore()")

    def translate(self, x, y):
        if self.show_low:
            print("    translate(", x, y, ")")

    def scale(self, xscale, yscale):
        if self.show_low:
            print("    scale(", xscale, yscale, ")")

    def set_transform(self, x, y, z, a, b, c):
        if self.show_low:
            print("    set_transform(", x, y, z, a, b, c, ")")

    def rotate(self, angle):
        if self.show_low:
            print("    rotate(", angle, ")")

    def begin_path(self):
        if self.show_low:
            print("    begin_path()")

    def close_path(self):
        if self.show_low:
            print("    close_path()")

    def ellipse(self, x, y, radiusX, radiusY, a, b, angle):
        if self.show_low:
            print("    ellipse(", x, y, radiusX, radiusY, a, b, angle, ")")

    def put_image_data(self, scaled, x, y):
        if self.show_low:
            print("    put_image_data(", scaled, x, y, ")")

    def create_image_data(self, width, height):
        if self.show_low:
            print("    create_image_data(", width, height, ")")
