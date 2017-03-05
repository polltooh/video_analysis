from TensorflowToolbox.utility import file_io
import numpy as np
import cv2

ucsd_label_path = "/media/dog/data/UCSD/gtDensities/"
ucsd_image_path = "/media/dog/data/UCSD/images/"
mask_file_name = "/media/dog/data/UCSD/Mask/mask227.npy"

def file_list_to_train_list(file_list):
    file_list = [t + " " + t.replace(image_ext, desmap_ext).\
                            replace(ucsd_image_path, ucsd_label_path) + " " \
                    + mask_file_name for t in file_list]

    return file_list

if __name__ == "__main__":
    image_ext = ".jpg"
    desmap_ext = ".desmap"
    file_list_dir = "../file_list/"

    save_train_file_name = "ucsd_train_list1.txt"
    save_test_file_name = "ucsd_test_list1.txt"
    
    image_list = file_io.get_listfile(ucsd_image_path, image_ext)
    image_list.sort()

    file_list = file_list_to_train_list(image_list)
    
    train_list = file_list[600:1400]
    test_list = file_list[0:600] + file_list[1400:]
    
    file_io.save_file(train_list, file_list_dir + save_train_file_name, True)
    file_io.save_file(test_list, file_list_dir + save_test_file_name, True)

