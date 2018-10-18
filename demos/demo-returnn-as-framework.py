#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + "/..")  # parent dir for Returnn

import better_exchook
better_exchook.install()

from pprint import pprint
from TFEngine import Engine
from Dataset import init_dataset
from Config import get_global_config
from Util import get_login_username

config = get_global_config(auto_create=True)
config.update(dict(
  batching="random",
  batch_size=5000,
  max_seqs=10,
  chunking="0",

  network={
    "fw0": {"class": "rec", "unit": "NativeLstm2", "dropout": 0.1, "n_out": 10},
    "output": {"class": "softmax", "loss": "ce", "from": ["fw0"]}
  },

  # training
  nadam=True,
  learning_rate=0.01,
  num_epochs=100,
  debug_add_check_numerics_ops=True,

  model="/tmp/%s/returnn-demo-as-framework/model" % get_login_username(),
  cleanup_old_models=True,

  learning_rate_control="newbob_multi_epoch",
  learning_rate_control_relative_error_relative_lr=True,
  newbob_multi_num_epochs=3, newbob_multi_update_interval=1, newbob_learning_rate_decay=0.9,
  learning_rate_file="/tmp/%s/returnn-demo-as-framework/newbob.data" % get_login_username(),

  # log
  log_verbosity=3
))

engine = Engine(config)

train_data = init_dataset({"class": "Task12AXDataset", "num_seqs": 1000, "name": "train"})
dev_data = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "dev", "fixed_random_seed": 1})

engine.init_train_from_config(train_data=train_data, dev_data=dev_data)
engine.train()
