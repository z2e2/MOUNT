import torch
import numpy as np
from mount import *

from config import Config

opt = Config()

data = Data(opt.DATA_ROOT, opt.MAXLEN, opt.short_AMINO, dataset_type='train', train_val_split=opt.train_val_split)

train_target_data, val_target_data = data.load()

train_eval(opt, train_target_data, val_target_data)