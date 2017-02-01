#!/bin/bash

data_dir="../data"
echo $data_dir

python sample_image.py $data_dir
python msk2png.py $data_dir
python msk_resize.py $data_dir
python label_preprocess.py $data_dir
python ../gen_list_script/gen_train_list.py $data_dir
