from args import Args
from datargs import parse
import wandb
from models import train

# -- Sweep configs
sweep_config = {'method': 'grid', 'name': 'Tuning lr + scheduler'}
metric = {'name': 'f1_score_micro_valid', 'goal': 'maximize'}

parameters_dict = {
    # 'rotation_degree' : {'values' : [45, 90, 135, 180]},
    # 'label_smoothing' : {'values' : [1e-1,1e-2,1e-3,1e-4]},
    # 'decay' : {'values' : [1e-2,1e-3, 1e-4, 1e-5]},
    'model_name' : {'value' : 'unet'},
    'lr' : {'values' : [1e-3,1e-4]},
    'criterion' :   {'value' : "dice+bce"},
    'lr_scheduler': {'value': ["ExponentialLR","ReduceLROnPlateau","CosineAnnealingLR"]}, 
    'in_channels':  {'value':4},
    'batchsize':    {'value':32},
    'epochs':       {'value':15}
}
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict


def my_train_func():
    """function to help run sweeps.
    """
    args = parse(Args)

    if __name__ == '__main__':

        # -- Initialize wandb
        with wandb.init(config=sweep_config):

            config = wandb.config

            # -- Hyperparameters to tune
            # args.rotate_degree = config.rotation_degree
            # args.label_smoothing_epsilon = config.label_smoothing
            # args.weight_decay = config.decay

            args.model_name = config.model_name
            args.criterion = config.criterion
            args.lr = config.lr
            args.lr_scheduler = config.lr_scheduler
            args.in_channels = config.in_channels
            args.batch_size = config.batchsize
            args.max_epochs = config.epochs
            args.max_steps = -1


            train(args=args)


if False:  # set to 'True' to run
    sweep_id = wandb.sweep(
            sweep_config,
            entity='fadelmamar',
            project='ECEO')
    wandb.agent(sweep_id, function=my_train_func, count=6)
