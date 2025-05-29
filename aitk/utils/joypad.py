# -*- coding: utf-8 -*-
# ***********************************************************
# aitk.utils: Python utils for AI
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.utils
#
# ***********************************************************

try:
    from ipywidgets import GridspecLayout, Button, Layout, VBox
except ImportError:
    pass

import math

def control(button):
    print(button.description)

class JoyPad():
    """
    A dynamic joypad made up of buttons.

    Args:
        scale: the scaling of translate and rotate values; defaults (1.0, 1.0)
        width: width of the widget; default 250
        height: height of the widget; default 250
        function: the function to call when clicking a button; default print
    """
    def __init__(self, scale=(1.0, 1.0), width=250, height=250, function=print):
        self.width = "%spx" % width if width is not None else None
        self.height = "%spx" % height if height is not None else None
        self.cell_width = math.floor(width / 3)
        self.cell_height = math.floor(height / 3)
        self.function = function
        self.scale = scale
        self.arrows = [
            "⬉⬆⬈",
            "⬅⊙➡",
            "⬋⬇⬊",
        ]
        self.movement = [
            [(1.0, -1.), (1.0, 0.0), (1.0, 1.0)],
            [(0.0, -1.), (0.0, 0.0), (0.0, 1.0)],
            [(-1., -1.), (-1., 0.0), (-1., 1.0)],
        ]
        self.last_movement = (0, 0)
        self.grid = GridspecLayout(3, 3, width=self.width, height=self.height)
        for row in range(3):
            for col in range(3):
                layout = Layout(
                    width="%spx" % self.cell_width,
                    height="%spx" % self.cell_height,
                    max_width="%spx" % self.cell_width,
                    max_height="%spx" % self.cell_height,
                    margin="0px",
                )
                style = {
                    "font_size": "xx-large",
                }
                self.grid[row, col] = Button(description=self.arrows[row][col], layout=layout, style=style)
                self.grid[row, col].on_click(lambda button, row=row, col=col: self.control(row, col))

        self.continue_button = Button(description="Continue", layout={'width': self.width, 'height': "50px", "padding": "5px"}, style=style)
        self.continue_button.on_click(self.continue_motion)
        self.widget = VBox(
            children=[
                self.grid,
                self.continue_button,
            ]
        )
                                     
    def continue_motion(self, button):
        translate, rotate = self.last_movement
        self.function(
            translate * self.scale[0],
            rotate * self.scale[1],
        )

    def control(self, row, col):
        self.last_movement = self.movement[row][col]
        translate, rotate = self.last_movement
        self.function(
            translate * self.scale[0],
            rotate * self.scale[1],
        )

    def watch(self):
        display(self.widget)

    def get_widget(self):
        return self.widget
