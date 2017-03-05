import scipy.io as sio
from TensorflowToolbox.utility import file_io
import numpy as np
import cv2

dsize = (227, 227)

ucsd_label_path = "/media/dog/data/UCSD/gtDensities/"
ucsd_image_path = "/media/dog/data/UCSD/images/"
mask_file_name = "/media/dog/data/UCSD/Mask/vidf1_33_roi_mainwalkway.mat"
#ucsd_label_path = "/Users/Geoff/Documents/my_git/video_analysis/data_process_script/ucsd/desmap/"
#ucsd_image_path = "/Users/Geoff/Documents/my_git/video_analysis/data_process_script/ucsd/image/"
#mask_file_name = "/Users/Geoff/Documents/my_git/temp_data/vidf1_33_roi_mainwalkway.mat"


def proc_mask():
    mask = np.ones(dsize, np.float32)
    new_mask_name = mask_file_name.replace(".mat", ".npy")
    mask_name = new_mask_name
    mask.tofile(mask_name)

    #mat = sio.loadmat(mask_file_name)
    #mask = mat["roi"][0][0][2]
    #mask.tofile(new_mask_name)
    #mask = cv2.resize(mask, dsize, cv2.INTER_NEAREST)
    #print(mask)
    #cv2.imshow("mask", mask * 255)
    #cv2.waitKey(0)
   
def proc_image():
    image_list = file_io.get_listfile(ucsd_image_path, image_ext)
    for img_name in image_list:
        img = cv2.imread(img_name)
        new_image_name = img_name.replace(image_ext, ".jpg")
        cv2.imwrite(new_image_name, img)

def proc_label():
    label_list = file_io.get_listfile(ucsd_label_path, label_ext)
    for l_name in label_list:
        mat = sio.loadmat(l_name)
        desmap = mat["gtDensities"]
        desmap = desmap.astype(np.float32)  
        new_label_name = l_name.replace(label_ext, new_label_ext)
        desmap.tofile(new_label_name)

if __name__ == "__main__":
    image_ext = ".png"
    label_ext = ".png_gtDen.mat"
    new_label_ext = ".desmap"
    proc_mask()
    proc_image()
    proc_label()



