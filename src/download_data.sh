#!/bin/bash

save_path="../data/images-2018"
sample_points_path="../data/brasil_coverage_2018_sample.csv"
sensor="S2"
start_date="2018-01-01"
end_date="2018-12-31"
period="12M"
buffer=750

source ../.venv/bin/activate

python download_data.py   --save_path $save_path --sample_points_path $sample_points_path\
    --sensor $sensor\
    --start_date $start_date --end_date $end_date\
    --num_workers 32 \
    --period_length $period\
    --buffer $buffer\
    --processing_level "TOA" \

deactivate