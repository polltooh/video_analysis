from TensorflowToolbox.utility import file_io
import numpy as np
import scipy.io as sio
import cv2

#laebl_path = "/media/dog/data/WorldExpo/train_label/"


CV_VERSION = cv2.__version__.split(".")[0]

dsize = (576, 720)
new_dsize = (256, 256)

def gen_mask(mask_name):
    mat = sio.loadmat(mask_name)
    y_coord = mat["maskVerticesYCoordinates"]
    x_coord = mat["maskVerticesXCoordinates"]
    
    mask_pts = np.hstack((x_coord, y_coord))
    pts = np.array(mask_pts, np.int32)
    h, w = dsize 
    img = np.zeros((h, w), np.float32)

    if CV_VERSION == '3':
        img = cv2.fillPoly(img, [pts], (1, 1))
    elif CV_VERSION == '2':
        cv2.fillPoly(img, [pts], (1, 1))

    img = cv2.resize(img, new_dsize,interpolation= cv2.INTER_NEAREST)
    _, img = cv2.threshold(img, 0, 1, 0)

    return img 

def gen_desmap(desmap_name):
    mat = sio.loadmat(desmap_name)
    desmap = mat["gtDensity"]
    h, w = desmap.shape
    resized_desmap = cv2.resize(desmap, new_dsize)
    if np.sum(resized_desmap) == 0:
        scale = 1
    else:
        scale = np.sum(desmap) / np.sum(resized_desmap)
    resized_desmap *= scale
    #print(np.sum(desmap))
    #print(np.sum(resized_desmap))

    return resized_desmap

def gen_img(img_name):
    img = cv2.imread(img_name)
    img = cv2.resize(img, new_dsize)
    return img


if __name__ == "__main__":
    for i in range(2):
        if i == 0:
            label_path = "/media/dog/data/WorldExpo/train_label/"
            mask_dir = "/media/dog/data/WorldExpo/mask/"
            desmap_dir = "/media/dog/data/WorldExpo/gtDensity/gtDensity/"
            img_dir = "/media/dog/data/WorldExpo/train_frame/"
        else:
            label_path = "/media/dog/data/WorldExpo/test_label/"
            mask_dir = "/media/dog/data/WorldExpo/test_mask/"
            desmap_dir = "/media/dog/data/WorldExpo/test_gtDensity/test_gtDensity/"
            img_dir = "/media/dog/data/WorldExpo/test_frame/"

        label_list = file_io.get_dir_list(label_path)
        scale_str = str(new_dsize[0])
        for label_l in label_list:
            mask_name = label_l + "/roi.mat"
            new_name = mask_dir + '/' + label_l.split("/")[-1] + "_mask_" + \
                            scale_str+ ".npy"
            mask = gen_mask(mask_name)
            mask.tofile(new_name)
        
        desmap_list = file_io.get_listfile(desmap_dir, ".mat")
        for d_name in desmap_list:
            desmap = gen_desmap(d_name)
            new_name = d_name.replace(".jpg_gtDen.mat", "_" + scale_str + ".npy")
            desmap.tofile(new_name)

        img_list = file_io.get_listfile(img_dir, ".jpg")
        for img_name in img_list:
            if not img_name.endswith("_" + scale_str + ".jpg"):
                img = gen_img(img_name)
                new_img_name = img_name.replace(".jpg", "_" + scale_str + ".jpg")
                cv2.imwrite(new_img_name, img)









