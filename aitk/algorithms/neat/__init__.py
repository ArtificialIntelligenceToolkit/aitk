"""A NEAT (NeuroEvolution of Augmenting Topologies) implementation"""

# Either do this, or add them in setup.py:
import aitk.algorithms.neat.nn as nn
import aitk.algorithms.neat.ctrnn as ctrnn
import aitk.algorithms.neat.iznn as iznn
import aitk.algorithms.neat.distributed as distributed
import aitk.algorithms.neat.visualize as visualize

from .config import Config
from .population import Population, CompleteExtinctionException
from .genome import DefaultGenome
from .reproduction import DefaultReproduction
from .stagnation import DefaultStagnation
from .reporting import StdOutReporter
from .species import DefaultSpeciesSet
from .statistics import StatisticsReporter
from .parallel import ParallelEvaluator
from .distributed import DistributedEvaluator, host_is_local
from .threaded import ThreadedEvaluator
from .checkpoint import Checkpointer
