# -*- coding: utf-8 -*-
# *************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2020 Calysto Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
#
# *************************************


class Hit:
    def __init__(
            self, wall, robot, height, x, y, distance, color, start_x, start_y, boundary, angle
    ):
        self.wall = wall
        self.robot = robot
        self.height = height
        self.x = x
        self.y = y
        self.distance = distance
        self.color = color
        self.start_x = start_x
        self.start_y = start_y
        self.boundary = boundary
        self.angle = angle

    def __repr__(self):
        return "<Hit(%s,%s) distance=%s, height=%s>" % (
            self.x,
            self.y,
            self.distance,
            self.height,
        )
