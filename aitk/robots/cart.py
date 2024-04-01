# -*- coding: utf-8 -*-
# *************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2020 Calysto Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
#
# *************************************

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

Python version based on:
https://raw.githubusercontent.com/openai/gym/master/gym/envs/classic_control/cartpole.py
"""

import math
import random

from .world import World, Robot
from .utils import rotate_around

from aitk.utils import Color

BLUE = Color("blue")
BLACK = Color("black")
YELLOW = Color(207, 166, 54)
GREY = Color("grey")

class Cart(Robot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.kinematics_integrator = 'euler'
        self.reset()

    def reset(self):
        # cart_state is: (x, x_dot, theta, theta_dot)
        self.cart_state = [(0.05 - random.random() * 0.1) for i in range(4)]
        self.action = 0
        self._set_pose(0, 0, 0)
        self.reward = 0

    def move(self, action):
        self.action = action

    def _step(self, time_step):
        # action is 0, 1 or -1
        x, x_dot, theta, theta_dot = self.cart_state
        if self.action == 1:
            force = self.force_mag
        elif self.action == -1:
            force = -self.force_mag
        else:
            force = 0
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.world.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + time_step * x_dot
            x_dot = x_dot + time_step * xacc
            theta = theta + time_step * theta_dot
            theta_dot = theta_dot + time_step * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + time_step * xacc
            x = x + time_step * x_dot
            theta_dot = theta_dot + time_step * thetaacc
            theta = theta + time_step * theta_dot

        self.x = x
        self.tvx = x_dot
        self.a = theta
        self.tva = theta_dot
        self.cart_state = (x, x_dot, theta, theta_dot)

    def get_observation(self):
        return self.cart_state

    def update(self, draw_list=None):
        pass

    def draw(self, backend):
        cart_x = self.x * 10
        backend.strokeStyle(GREY, 1)
        backend.draw_line(0, self.world.height/2, self.world.width, self.world.height/2)
        backend.set_fill_style(BLACK)
        backend.draw_rect(self.world.width/2 + cart_x - 25,
                          self.world.height/2 - 10,
                          50,
                          20)
        # draw pole
        backend.strokeStyle(YELLOW, 5)
        x, y = rotate_around(self.world.width/2 + cart_x,
                             self.world.height/2,
                             100, self.cart_state[2] - math.pi/2)
        backend.draw_line(self.world.width/2 + cart_x,
                          self.world.height/2,
                          x, y)
        backend.set_fill_style(BLUE)
        backend.strokeStyle(BLUE, 1)
        backend.draw_circle(self.world.width/2 + cart_x, self.world.height/2, 5)


class CartWorld(World):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        -1    Push cart to the left
         1    Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    """
    def __init__(self, *args, **kwargs):
        kwargs["boundary_wall"] = False
        kwargs["ground_color"] = "white"
        super().__init__(*args, **kwargs)
        self.time_step = 0.02
        self._time_decimal_places = 2
        self.gravity = 9.8
        self.cart = Cart(0)
        self.add_robot(self.cart)
        self.cart.reset()
