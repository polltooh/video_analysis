import numpy as np
import cv2
from TensorflowToolbox.utility import file_io

desmap_dir = "/media/dog/data/UCSD/gtDensities/"

desmap_list = file_io.get_listfile(desmap_dir, "desmap")
new_list = list()
for f in desmap_list:
    num = np.sum(np.fromfile(f, np.float32))
    format_num = "%.2f"%num
    new_list.append(f + " " + format_num)
    

new_list.sort()
file_io.save_file(new_list, "ucsd_file_label_num.txt")
