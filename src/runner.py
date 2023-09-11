CUDA_LAUNCH_BLOCKING="1"
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 11:29:22 2022

@author: SeydouF.

This script is the runner file used to launch the training routine.
"""
from args import Args
from datargs import parse
import wandb
from models import train
from utils import get_level_indices

args = parse(Args)

# -- Debugging
# args.debug_mode = False #-- enabling/disabling debugging mode
if args.debug_mode:
    args.tag = ("Debug",)
    args.batch_size = 8
    args.optimize_threshold = True
    args.precision = 32


if __name__ == '__main__':

    # get number of classes to predict
    # it is done for logging to wandb but it's **optional**
    if args.model_name == 'baseline':
        indices_level1 ,indices_level2,indices_level3, indices_level4,indices_leaf = get_level_indices(args.label_hierarchy)
        assert args.baseline_level in ['level1','level2','level3','level4','leaf'],'provide a correct input'
        levels_indices = dict(zip(['level1','level2','level3','level4','leaf'],
                                        [indices_level1 ,indices_level2,
                                        indices_level3, indices_level4,
                                        indices_leaf]))
        args.num_classes = levels_indices[args.baseline_level].shape[0]
    
    # -- Initialize logger
    wandb.init(config=args,
               project=args.project_name,
               name=args.run_name,
               entity='fadelmamar',
               tags=args.tag)

    train(args=args)

    wandb.finish()  # -- End
