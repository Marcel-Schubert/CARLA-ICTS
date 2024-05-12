import sys

from BiTrap.bitrap.utils.scheduler import ParamScheduler, sigmoid_anneal
from ped_path_predictor.bitrap_wrapper import BiTrapWrapper

sys.path.append('./BiTrap')

from BiTrap.bitrap.modeling.bitrap_np import BiTraPNP
from BiTrap.configs import cfg
from P3VI.utils import load_data
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import torch



if __name__ == "__main__":
    bitrap = BiTrapWrapper()
    bitrap.train()
