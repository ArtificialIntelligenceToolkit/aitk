# -*- coding: utf-8 -*-
# ************************************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
# ************************************************************

from ..config import get_backend


def make_backend(width, height, scale):
    BACKEND, ARGS = get_backend()

    if BACKEND == "canvas":
        try:
            from .canvas import CanvasBackend
            
            return CanvasBackend(
                width=round(width * scale),
                height=round(height * scale),
                sync_image_data=True,
                **ARGS
            )
        except Exception:
            print("Failed to make canvas backend")
            return None
    elif BACKEND == "svg":
        try:
            from .svg import SVGBackend
            
            return SVGBackend(width, height, scale, **ARGS)
        except Exception:
            print("Failed to make svg backend")
            return None        
    elif BACKEND == "pil":
        try:
            from .pil import PILBackend
            
            return PILBackend(width, height, scale, **ARGS)
        except Exception:
            print("Failed to make pil backend")
            return None
    elif BACKEND == "debug":
        from .debug import DebugBackend

        return DebugBackend(width, height, scale, **ARGS)
    else:
        raise ValueError("unknown backend type: %r" % BACKEND)
