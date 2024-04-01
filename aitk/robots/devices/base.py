# -*- coding: utf-8 -*-
# ************************************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
# ************************************************************

class BaseDevice():
    def verify_config(self, valid_keys, config):
        config_keys = set(list(config.keys()))
        extra_keys = config_keys - valid_keys

        if len(extra_keys) > 0:
            raise TypeError("invalid key(s) for device: %r" % extra_keys)
