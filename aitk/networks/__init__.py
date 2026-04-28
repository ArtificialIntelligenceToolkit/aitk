# -*- coding: utf-8 -*-
# ******************************************************
# aitk.networks: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.networks
#
# ******************************************************

import io
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Suppress absl/CUDA/protobuf noise at both the fd and Python levels
_devnull = os.open(os.devnull, os.O_WRONLY)
_old_stderr_fd = os.dup(2)
_old_sys_stderr = sys.stderr
os.dup2(_devnull, 2)
os.close(_devnull)
sys.stderr = io.StringIO()
try:
    from tensorflow.keras.layers import *
    from .network import Network, SequentialNetwork, SimpleNetwork  # noqa: F401
finally:
    sys.stderr = _old_sys_stderr
    os.dup2(_old_stderr_fd, 2)
    os.close(_old_stderr_fd)
