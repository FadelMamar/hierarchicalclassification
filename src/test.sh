#!/bin/bash
set -m

tag="test"
encoder_name='efficientnet-b3'
model_name="baseline" #"" "default" "baseline" "customArch"
in_channels=4
batch_size=32
customArchactivation="identity"
weights_path="" #Give path to weights
criterion="ce"
baseline_level="leaf" # states on which labels are used to compute evaluation metrics

source ../.venv/bin/activate

python runner_test.py  --encoder-name $encoder_name --model-name $model_name --criterion $criterion --baseline-level $baseline_level --in-channels $in_channels\
    --run-name "$model_name/$criterion"\
    --batch-size $batch_size\
    --apply-augmentation --disable-media-logging --tag $tag --checkpoint-path $weights_path

# python runner_test.py  --encoder-name $encoder_name --model-name $model_name --criterion $criterion --in-channels $in_channels\
#     --run-name "$model_name/$criterion/strategy:2/$customArchactivation"\
#     --batch-size $batch_size\
#     --apply-augmentation --disable-media-logging --tag $tag\
#     --customArch-strategy 2 --customArch-activation $customArchactivation\
#     --checkpoint-path $weights_path --customArch-useOtherLosses 

deactivate