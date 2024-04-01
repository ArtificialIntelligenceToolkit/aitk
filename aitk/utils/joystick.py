# -*- coding: utf-8 -*-
# ***********************************************************
# aitk.utils: Python utils for AI
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.utils
#
# ***********************************************************

import math

try:
    from ipywidgets import (
        Button,
        Layout,
        GridBox,
        ButtonStyle,
        FloatSlider,
        HBox,
        VBox,
        GridspecLayout,
        TwoByTwoLayout,
    )
    _has_ipywidgets = True
except ImportError:
    _has_ipywidgets = False
    
try:
    from ipycanvas import Canvas, hold_canvas
    _has_ipycanvas = True
except ImportError:
    _has_ipycanvas = False


def has_ipycanvas():
    return _has_ipycanvas

def has_ipywidgets():
    return _has_ipywidgets

class Joystick():
    def __init__(self, scale=(1.0, 1.0), width=250, height=250, function=print):
        """
        A dynamic joystick drawn on a Canvas.

        Args:
            scale: the scaling of translate and rotate values; defaults (1.0, 1.0)
            width: width of the widget; default 250
            height: height of the widget; default 250
            function: the function to call when changing joystick; default print
        """
        self.translate_scale = scale[0]
        self.rotate_scale = scale[1]
        self.last_translate = 0
        self.last_rotate = 0
        self.state = "up"
        self.width = width
        self.height = height
        self.function = function
        self.canvas = Canvas(width=self.width, height=self.height)
        self.canvas.on_mouse_move(self.handle_mouse_move)
        self.canvas.on_mouse_down(self.handle_mouse_down)
        self.canvas.on_mouse_up(self.handle_mouse_up)
        self.canvas.layout.min_width = "%spx" % self.width
        self.canvas.layout.max_height = "%spx" % self.height
        # Draw blank joystick:
        self.reset()

    def clear(self):
        self.canvas.fill_style = "#B0C4DE"
        self.canvas.fill_rect(0, 0, self.width, self.height)
        self.canvas.stroke_style = "black"
        self.canvas.stroke_rect(0, 0, self.width, self.height)

    def rotate_around(self, x, y, length, angle):
        return (x + length * math.cos(-angle),
                y - length * math.sin(-angle))

    def handle_mouse_move(self, x, y):
        if self.state == "down":
            self.last_translate = ((self.height/2 - y) / (self.height/2)) * self.translate_scale
            self.last_rotate = ((self.width/2 - x) / (self.width/2)) * self.rotate_scale
            self.function(self.last_translate, self.last_rotate)
            with hold_canvas(self.canvas):
                self.clear()
                self.canvas.stroke_style = "black"
                angle = math.atan2(self.width/2 - x, self.height/2 - y)
                x1, y1 = self.rotate_around(self.width/2, self.height/2, 10, -angle)
                x2, y2 = self.rotate_around(self.width/2, self.height/2, -10, -angle)
                points = [
                    (self.width/2, self.height/2),
                    (x1, y1),
                    (x, y),
                    (x2, y2),
                ]
                self.canvas.fill_style = "gray"
                self.canvas.fill_polygon(points)
                self.canvas.fill_circle(self.width/2, self.height/2, 10)
                self.canvas.fill_style = "black"
                self.canvas.fill_circle(x, y, self.width/10)

    def handle_mouse_down(self, x, y):
        self.state = "down"
        self.handle_mouse_move(x, y)

    def handle_mouse_up(self, x, y):
        self.state = "up"
        self.last_translate = 0
        self.last_rotate = 0
        self.function(0, 0)
        self.reset()

    def reset(self):
        # translate, rotate
        self.last_translate = 0
        self.last_rotate = 0
        self.function(0, 0)
        self.clear()
        self.canvas.fill_style = "black"
        self.canvas.fill_circle(self.width/2, self.height/2, self.width/10)

    def watch(self):
        display(self.canvas)

    def get_widget(self):
        return self.canvas


class NoJoystick():
    def __init__(self, scale=(1.0, 1.0), width=250, height=250, function=print):
        """
        A simple joystick via buttons and sliders.

        Args:
            scale: the scaling of translate and rotate values; defaults (1.0, 1.0)
            function: the function to call when changing joystick; default print
            width: width of the widget (ignored)
            height: height of the widget (ignored)
        """
        self.translate_scale = scale[0]
        self.rotate_scale = scale[1]
        self.last_translate = 0
        self.last_rotate = 0

        self.function = function
        self.arrows = [
            "⬉ ⬆ ⬈",
            " ╲｜╱ ",
            "⬅－⊙－➡",
            " ╱｜╲ ",
            "⬋ ⬇ ⬊",
        ]

        self.movement = [
            [(1.0, -1.), (1.0, -.5), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)],
            [(0.5, -1.), (0.5, -.5), (0.5, 0.0), (0.5, 0.5), (0.5, 1.0)],
            [(0.0, -1.), (0.0, -.5), (0.0, 0.0), (0.0, 0.5), (0.0, 1.0)],
            [(-.5, -1.), (-.5, -.5), (-.5, 0.0), (-.5, 0.5), (-.5, 1.0)],
            [(-1., -1.), (-1., -.5), (-1., 0.0), (-1., 0.5), (-1., 1.0)],
        ]

        # Make the widgets:
        layout = Layout(height="35px", width="35px")
        self.buttons = []
        for row in range(5):
            for col in range(5):
                button = Button(description=self.arrows[row][col], layout=layout)
                button.on_click(self.create_move(row, col))
                self.buttons.append(button)

        self.rotate_slider = FloatSlider(
            min=-1, max=1, step=0.1,
            continuous_update=True,
            readout=False,
            layout=Layout(width="200px", height="30px"))

        self.translate_slider = FloatSlider(
            min=-1, max=1, step=0.1,
            orientation="vertical",
            continuous_update=True,
            readout=False,
            layout=Layout(width="30px"))

        self.array = GridBox(
            children=self.buttons,
            layout=Layout(
                grid_template_rows='40px 40px 40px 40px 40px',
                grid_template_columns='40px 40px 40px 40px 40px',
                grid_gap='0px 0px',
                overflow="intial",
            )
        )

        self.controls = TwoByTwoLayout(
            top_left=self.translate_slider,
            top_right=self.array,
            bottom_right=self.rotate_slider,
            width="min-content",
        )

        self.translate_slider.observe(self.on_translate_change,
                                      names='value')
        self.rotate_slider.observe(self.on_rotate_change,
                                   names='value')

    def on_translate_change(self, change):
        self.last_translate = change['new'] * self.translate_scale
        self.function(self.last_translate, None)

    def on_rotate_change(self, change):
        self.last_rotate = -change['new'] * self.rotate_scale
        self.function(None, self.last_rotate)

    def create_move(self, row, col):
        def on_click(button):
            translate, rotate = self.movement[row][col]
            self.translate_slider.value = translate
            self.rotate_slider.value = rotate
        return on_click

    def watch(self):
        display(self.controls)

    def get_widget(self):
        return self.controls
