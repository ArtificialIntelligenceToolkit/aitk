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
    Output,
)
from typing import List, Tuple, Any, Optional


def move(
    world: World, robot: Robot, steering: float, power: float, seconds: float
) -> None:
    """
    Move the robot in the world based on steering and power inputs.

    This function updates the robot's movement, records sensor data and control inputs
    to the global data list, and advances the world simulation.

    Args:
        world: The simulation world containing the robot
        robot: The robot to control
        steering: Steering control value between -1 (left) and 1 (right)
        power: Power/throttle control value between 0 (stop) and 1 (full speed)
        seconds: Duration of the movement step in seconds

    Note:
        The function records sensor readings and control inputs at 10Hz (every 0.1 seconds)
        during the movement period. The robot's velocity is displayed as debug output.
    """
    translate = power
    rotate = steering

    # Record sensor data and control inputs at 10Hz during movement
    for i in range(int(seconds * 10)):
        # Get distances from all range sensors on the robot
        sensor_distances = robot.get_distances(scaled=True)
        control_inputs = (translate, -rotate)
        data.append((sensor_distances, control_inputs))

    # Move the robot with the specified controls
    robot.move(translate, -rotate)

    # Advance the world simulation
    with Output():
        world.seconds(seconds, real_time=False)

    # Display current velocity as debug output
    robot.speak("T: %.02f, R: %.02f" % robot.get_velocity())


def dashboard(
    world: World, robot: Robot, data: List[Tuple[List[float], Tuple[float, float]]]
) -> VBox:
    """
    Create an interactive dashboard widget for controlling a robot in a world.

    This function creates a Jupyter widget interface with controls for steering,
    power, and simulation timing, along with a visual representation of the world.

    Args:
        world: The simulation world containing the robot
        robot: The robot to control
        data: The list to store movement history

    Returns:
        VBox: A Jupyter widget containing the control interface and world display

    The dashboard includes:
        - Steering wheel slider (-1 to 1)
        - Gas/power slider (0 to 1)
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
        move(world, robot, steering.value, power.value, seconds.value)

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
        min=0,
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
