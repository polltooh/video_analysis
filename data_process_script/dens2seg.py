from TensorflowToolbox.utility import file_io
import numpy as np
import cv2
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "../data"

    dsize = (256, 256)
    data_ext = "_resize.jpg"
    label_ext = "_resize.desmap"
    seg_ext = "_resize.segmap"
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
