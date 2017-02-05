from TensorflowToolbox.utility import file_io
from TensorflowToolbox.utility import image_utility_func
import numpy as np
import cv2
import sys

dsize = (256, 256)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "../data"

    mask_dir_list = file_io.get_dir_list(data_dir)
    for mask_dir in mask_dir_list:
        mask_list = file_io.get_listfile(mask_dir, ".msk")
        for mask in mask_list:
            mask_img_name = mask.replace("msk", "png")
            mask_img = cv2.imread(mask_img_name, 0)
            bbox = image_utility_func.get_bbox(mask_img, 127)
            mask_img = mask_img[bbox[1]: bbox[1] + bbox[3],
                       bbox[0]: bbox[0] + bbox[2]]
            mask_img = cv2.resize(mask_img, dsize)
            mask_img = mask_img / 255
            mask_img = mask_img.astype(np.float32)
            mask_img[mask_img > 0] = 1.0
            np_name = mask_img_name.replace(".png", "_msk_resize.npy")

            mask_img.tofile(np_name)
