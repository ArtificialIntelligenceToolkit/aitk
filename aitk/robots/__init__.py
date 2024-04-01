# -*- coding: utf-8 -*-
# *************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2020 Calysto Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
#
# *************************************

from .config import setup_backend, switch_backend  # noqa: F401
from .devices import Camera, GroundCamera, LightSensor, RangeSensor, SmellSensor, Compass  # noqa: F401
from .robot import Robot, Scribbler, Vehicle  # noqa: F401
from .utils import gallery, load_world  # noqa: F401
from .world import Beacon, Bulb, Wall, World  # noqa: F401

setup_backend()  # checks os.environ

import time

# Test to see if time.sleep() works:
start_time = time.monotonic()
time.sleep(0.1)
if time.monotonic() - start_time < 0.1:
    print("time.sleep() failed; redefined")
    # It didn't work, we replace time.sleep:
    # happens with current Pyodide
    def sleep(seconds):
        start_time = time.monotonic()
        while time.monotonic() - start_time < seconds:
            pass
    time.sleep = sleep
