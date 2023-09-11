#!/bin/bash
set -m

tag="trainloaderShuffle" # "pre-trained"
encoder_name='efficientnet-b3'
# model_name can be: "default" "baseline" "customArch"
baseline_level="leaf" # states on which labels are used to compute evaluation metrics
epochs=15
max_steps=-1
in_channels=4
batch_size=32 #default is 32
sch="MultiplicativeLR" # "default" MultiplicativeLR
lr=0.0001 # rescaling lr according to defaultbatchsize?
T_0=15
exp_lr_gamma=0.85 # from 1e-4 to 1e-5 in T_0 epochs
# decays=(0.00001)
customArchactivation="identity" # 'RELU', 'SELU'
pretrained_weights="" # Give path to weights, if empty it's ignore

source ../.venv/bin/activate

# Baseline
python runner.py  --encoder-name $encoder_name --model-name "baseline" --baseline-level $baseline_level\
    --in-channels $in_channels\
    --criterion "ce" --lr $lr --lr-scheduler $sch --exp-lr-gamma $exp_lr_gamma --T-0 $T_0\
    --run-name "baseline/ce"\
    --max-epochs $epochs\
    --max-steps $max_steps\
    --batch-size $batch_size\
    --apply-augmentation --disable-media-logging --tag $tag\

# Global hierarchical models
criterion=("mcloss" "dice+bce" "bce" "focal" "treeminloss" "treemin+triplet")
for loss in "${criterion[@]}"
do
python runner.py  --encoder-name $encoder_name --model-name "default" --baseline-level $baseline_level\
    --in-channels $in_channels\
    --criterion $loss --lr $lr --lr-scheduler $sch --exp-lr-gamma $exp_lr_gamma --T-0 $T_0\
    --run-name "default/$loss"\
    --max-epochs $epochs\
    --max-steps $max_steps\
    --batch-size $batch_size\
    --apply-augmentation --disable-media-logging --tag $tag\
    --pretrained-encoder-weights $pretrained_weights 

python runner.py  --encoder-name $encoder_name --model-name "customArch"  --baseline-level $baseline_level --in-channels $in_channels\
    --criterion $loss --lr $lr --lr-scheduler $sch --exp-lr-gamma $exp_lr_gamma --T-0 $T_0\
    --run-name "customArch/$loss/strategy:2/$customArchactivation"\
    --max-epochs $epochs\
    --max-steps $max_steps\
    --batch-size $batch_size\
    --apply-augmentation --disable-media-logging --tag $tag\
    --customArch-strategy 2 --customArch-activation $customArchactivation --customArch-useOtherLosses\
    --pretrained-encoder-weights $pretrained_weights 

python runner.py  --encoder-name $encoder_name --baseline-level $baseline_level --model-name "customArch"  --in-channels $in_channels\
    --criterion $loss --lr 0.001 --lr-scheduler $sch --exp-lr-gamma $exp_lr_gamma --T-0 $T_0\
    --run-name "customArch/$loss/strategy:1/$customArchactivation"\
    --max-epochs $epochs\
    --max-steps $max_steps\
    --batch-size $batch_size\
    --apply-augmentation --disable-media-logging --tag $tag\
    --customArch-strategy 1 --customArch-activation $customArchactivation --customArch-useOtherLosses 

done

# Local hierarchical models
criterion=("ce" "dice+ce")
for loss in "${criterion[@]}"
do
python runner.py  --encoder-name $encoder_name --model-name "customArch"  --baseline-level $baseline_level --in-channels $in_channels\
    --criterion $loss --lr $lr --lr-scheduler $sch --exp-lr-gamma $exp_lr_gamma --T-0 $T_0\
    --run-name "customArch/$loss/strategy:2/$customArchactivation"\
    --max-epochs $epochs\
    --max-steps $max_steps\
    --batch-size $batch_size\
    --apply-augmentation --disable-media-logging --tag $tag\
    --customArch-strategy 2 --customArch-activation $customArchactivation\
    --pretrained-encoder-weights $pretrained_weights

python runner.py  --encoder-name $encoder_name --baseline-level $baseline_level --model-name "customArch"  --in-channels $in_channels\
    --criterion $loss --lr 0.001 --lr-scheduler $sch --exp-lr-gamma $exp_lr_gamma --T-0 $T_0\
    --run-name "customArch/$loss/strategy:1/$customArchactivation"\
    --max-epochs $epochs\
    --max-steps $max_steps\
    --batch-size $batch_size\
    --apply-augmentation --disable-media-logging --tag $tag\
    --customArch-strategy 1 --customArch-activation $customArchactivation\
    --pretrained-encoder-weights $pretrained_weights

done
deactivate