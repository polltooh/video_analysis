#!/bin/bash

data_dir="../data"
echo $data_dir

python3 sample_image.py $data_dir
python3 msk2png.py $data_dir
python3 msk_resize.py $data_dir
python3 label_preprocess.py $data_dir


