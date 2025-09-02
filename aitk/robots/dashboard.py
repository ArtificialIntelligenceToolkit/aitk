# -*- coding: utf-8 -*-
# ***********************************************************
# aitk.utils: Python utils for AI
#
# Copyright (c) 2025 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.utils
#
# ***********************************************************

from . import World, Robot

from ipywidgets import (
    HBox,
    VBox,
    Layout,
    FloatSlider,
    Label,
    SliderStyle,
    Button,
)
from typing import List, Tuple, Callable


def dashboard(
    world: World,
    robot: Robot,
    data: List[Tuple[List[float], Tuple[float, float]]],
    move: Callable,
) -> VBox:
    """
    Create an interactive dashboard widget for controlling a robot in a world.

    This function creates a Jupyter widget interface with controls for steering,
    power, and simulation timing, along with a visual representation of the world.

    Args:
        world: The simulation world containing the robot
        robot: The robot to control
        data: The list to store movement history
        move: A callable that executes robot movement. Should have the signature
            move(world, robot, steering, power, seconds), where it updates the robot's
            state in the world according to the given steering, power, and time step.

    Returns:
        VBox: A Jupyter widget containing the control interface and world display

    The dashboard includes:
        - Steering wheel slider (-1 to 1)
        - Gas/power slider (-1 to 1)
        - Step button to execute movement
        - Seconds per step slider (0.1 to 5.0)
        - Visual world display widget
    """
    # Reset world and clear data
    world.reset()
    data.clear()

    # Create centered layout for labels
    center_layout = Layout(display="flex", justify_content="center")

    # Create step button
    step = Button(
        description="Step",
        layout=Layout(
            display="flex",
            justify_content="center",
            align_items="center",
            width="99%",
        ),
    )

    # Define step button click handler
    def step_click(button: Button) -> None:
        """Handle step button click by executing robot movement."""
        move(world, robot, data, steering.value, power.value, seconds.value)

    step.on_click(step_click)
    step.style.button_color = "lightblue"

    # Create control label
    label = Label(value="Robot Control", layout=center_layout)

    # Create steering control slider
    steering = FloatSlider(
        description="Steering wheel:",
        min=-1,
        max=1,
        value=0,
        layout=Layout(width="400px"),
        style=SliderStyle(description_width="100px"),
    )

    # Create power control slider
    power = FloatSlider(
        description="Gas:",
        min=-1,
        max=1,
        value=0,
        layout=Layout(width="400px"),
        style=SliderStyle(description_width="100px"),
    )

    # Create timing control slider
    seconds = FloatSlider(
        description="Seconds per step:",
        min=0.1,
        max=5,
        value=0.1,
        layout=Layout(width="400px"),
        style=SliderStyle(description_width="200px"),
    )

    # Create main layout
    layout = Layout()
    controls = VBox(
        children=[
            HBox(
                children=[
                    VBox(
                        [label, steering, power, step],
                        layout=Layout(border="solid 2px"),
                    ),
                    world.get_widget(width=500),
                ]
            ),
            seconds,
        ],
        layout=layout,
    )

    return controls
