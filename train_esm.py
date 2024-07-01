"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
import argparse
import os
os.environ['HF_HOME'] = '/data1/mingjia/cache/'
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import proteinchat.tasks as tasks
from proteinchat.common.config import Config
from proteinchat.common.dist_utils import get_rank, init_distributed_mode
from proteinchat.common.logger import setup_logger
from proteinchat.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from proteinchat.common.registry import registry
from proteinchat.common.utils import now

# imports modules for registration
from proteinchat.datasets.builders import *
from proteinchat.models import *
from proteinchat.runners import *
from proteinchat.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", default="configs/proteinchat_stage1.yaml", help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    job_id = now()
    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    runner.train()


if __name__ == "__main__":
    main()
