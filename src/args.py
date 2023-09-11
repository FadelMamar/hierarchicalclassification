from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import torch
import os


@dataclass
class Args:
    """This class saves the arguments used in the experiments. The arguments can be set here or through command line. when using the command line every underscore ('_') become a dash ('-').
    """

    # -- random seed
    seed: int = 41

    # -- wandb
    project_name: str = 'ECEO'  # 
    run_name: str = 'Debug'
    tag: Sequence[str] = ('V0')

    # data location
    data_dir : Path = Path("../data/images-2018")
    data_info : Path = Path("../data/brasil_coverage_2018_sample.csv")
    label_hierarchy : Path = Path("../data/labelhierarchy.csv")
    label_names : Path = Path('../data/labelnames.csv')

    # -- data preparation
    debug_mode: bool = False
    pin_memory: bool = True  # dataloader
    num_workers: int = os.cpu_count() // 2
    use_mixup: bool = False  # at train time
    use_label_smoothing: bool = False
    label_smoothing_epsilon: float = 1e-4   
    mixup_alpha: float = 1e-2  # Tuned
    apply_augmentation: bool = False  # at train time

    augmentations: Sequence[str] = (
        "ShiftScaleRotate",
        "HorizontalFlip",
        "GaussianBlur")
        
    
    rotate_degree: float = 180.0  # Tuned
    resizing_mode: str = 'nearest'  # Tuned. Another option is "None" as a string

    traindata_mean: Sequence[float] = (4.2111, 954.6794, 994.5911)  # Ndvi -G - B 
    traindata_std: Sequence[float] = ( 45.5396, 512.6230, 423.9982) # Ndvi -G - B

    traindata_mean_4channels: Sequence[float] = (863.2990, 954.6794,994.5911, 2370.9519)  # R -G - B Nir
    traindata_std_4channels: Sequence[float] = (725.7848, 512.6230,423.9982, 939.1970)    # R -G - B Nir
    
    # See https://smp.readthedocs.io/en/latest/models.html
    model_name: str = 'default' # 'customArch' or 'default' 'baseline'
    encoder_name: str = 'efficientnet-b3' # 'efficientnetb3-pytorch', 'resnet18', or anther encoder name. See smp 
    
    output_path: Optional[Path] = None
    decoder_use_batchnorm: bool = True
    encoder_weights: str = "imagenet"
    encoder_depth: int = 5  # do not change
    in_channels: int = 4
    num_classes: int = 32 # number of classes predicted by the neural network
    num_labels: int = 32 # number of classes in the hierarchy
    input_size: int = 256  
    activation: Optional[str] = None  # for final convolution layer “sigmoid”, “softmax”, “logsoftmax”, “tanh”, “identity”, callable and None.
    decoder_attention_type = None
    pretrained_encoder_weights: str = '' # "./ECEO/0175tl5a/checkpoints/epoch=0-step=304.ckpt"
    freeze_encoder:bool = False
    save_weights_only: bool = True  # says if weights should be logged
    log_weights: bool = False
    checkpoint_path:str = ""
    

    # -- Training params
    lr: float = 1e-4  # 
    auto_tune_lr: bool = False
    optimizer: str = "Adam"  # or "SGD"
    batch_size: int = 32
    max_epochs: int = 30
    criterion: str = "dice+bce"  # "bce", "dice+bce" "mcloss"
    focal_loss_gamma: float = 2.0
    device: str = "gpu" if torch.cuda.is_available() else "cpu"
    max_steps: int = 10  # if set to another value then it will be the earliest between max_epochs and max_steps
    max_time = "00:12:00:00"  # stop after 12 hours
    weight_decay: float = 1e-5 # Tuned  
    optimize_threshold: bool = False
    prediction_threshold: float = 0.5  # -- default
    train_threshold_default: float = 0.5  # -- default (optimal threshold w.r.t F1-score)
    precision: int = 32
    focus_on_minority_class: bool = True
    baseline_level:str = 'leaf' #'level1','level2','level3','level4','leaf'

    # -- Lr scheduling
    metric_to_monitor_lr_scheduler: str = "valid_loss"
    lr_scheduler: str = "CosineAnnealingLR"  # Options are "CosineAnnealingWarmRestarts", "MultiplicativeLR", OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
    exp_lr_gamma: float = 0.85
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.1  # <1 !
    min_lr: float = 1e-5
    lr_scheduler_mode: str = 'min'  # goal is to minimize the metric to monitor
    T_0: int = 15 # peroid for decreasing lr
    T_mult: int = 1

    # -- Early stopping
    metric_to_monitor_early_stop: str = "valid_loss" #"f1_score_macro_valid"
    early_stopping_patience: int = 5
    early_stopping: bool = False
    early_stop_mode: str = "min"  # goal is to minimize the metric to monitor

    # -- logging
    log_pred_every_nstep: int = 1000
    log_pred_every_nepoch: int = 10
    log_saliency_map: bool = False
    colormap: str = 'viridis'
    saliency_map_method: str = 'gcam'  # 'ggcam', 'gcampp', 'gbp'
    attention_layer: str = 'segmentation_head.0'
    disable_media_logging: bool = False
    use_all_train_data:bool = False

    # hierarchy aware losses
    hierarchyloss_with_dice : bool = False
    hierarchyloss_with_focal: bool = False

    # custom architectures
    customArch_strategy:int = 2
    customArch_activation:str = 'identity'
    customArch_useOtherLosses:bool = False