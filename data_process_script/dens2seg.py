from TensorflowToolbox.utility import file_io
import numpy as np
import cv2
import os
import sys

if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Usage: label_preprocess.py data_dir size_len")
        exit(1)

    data_dir = sys.argv[1]
    size_len = int(sys.argv[2])
    dsize = (size_len, size_len)

    data_ext = "_%d.jpg"%size_len
    label_ext = "_%d.desmap"%size_len
    seg_ext = "_%d.segmap"%size_len

    cam_dir_list = file_io.get_dir_list(data_dir)

    train_list = list()
    test_list = list()
    full_file_list = list()
    for cam_dir in cam_dir_list:
        video_list = file_io.get_listfile(cam_dir, ".avi")
        for file_name in video_list:
            data_dir_name = file_name.replace(".avi", "")
            curr_data_list = file_io.get_listfile(data_dir_name, label_ext)
            for desmap_filename in curr_data_list:
                desmap = np.fromfile(desmap_filename, np.float32)
                desmap = np.reshape(desmap, dsize)
                segmap = desmap > 0
                segmap = segmap.astype(np.float32)
                segmap_filename = desmap_filename.replace(label_ext, seg_ext)
                segmap.tofile(segmap_filename)
                #cv2.imshow("image", segmap)
                #cv2.waitKey(0)
