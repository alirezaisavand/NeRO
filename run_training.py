import argparse

from train.trainer import Trainer
from utils.base_utils import load_cfg
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
flags = parser.parse_args()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
pointnerf_path = os.path.join(project_root, 'pointnerf')
sys.path.insert(0, pointnerf_path)

Trainer(load_cfg(flags.cfg)).run()
