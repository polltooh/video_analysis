from TensorflowToolbox.utility import file_io
import numpy as np
import scipy.io as sio
import cv2

trancos_data_path = "/Users/Geoff/Documents/my_git/data/Trancos/"

def mat_to_np(file_name):
    mat = sio.loadmat(file_name)
    desmap = mat["gtDensities"]
    desmap = desmap.astype(np.float32)  
    return desmap

    #cv2.imshow("image", desmap * 255)
    #cv2.waitKey(0)

def single_file_list(file_list_name):
    img_ext = ".jpg"
    mat_ext = "dots.png.mat"
    desmap_ext = ".desmap"
    mask_ext = "_mask.npy"
    img_size = (227, 227)
    file_list = file_io. read_file(file_list_name)
    file_list = [trancos_data_path + f for f in file_list]
    for f in file_list:
        mat_name = f.replace(img_ext, mat_ext)
        f_np = mat_to_np(mat_name)
        new_name = mat_name.replace(mat_ext, desmap_ext)
        f_np.tofile(new_name) 
        mask = np.ones(img_size, np.float32)
        mask_name = f.replace(img_ext, mask_ext)
        mask.tofile(mask_name)


if __name__ == "__main__":
    train_file_list_name = "trainval.txt"
    single_file_list(train_file_list_name)
    test_file_list_name= "test.txt"
    single_file_list(test_file_list_name)

