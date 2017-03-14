from TensorflowToolbox.utility import file_io
import cv2
import numpy as np

def norm_image(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def opencv_plot(desmap_name, mask):
    desmap = np.fromfile(desmap_name, np.float32)
    desmap = np.reshape(desmap, (114, 114))
    desmap *= 30.0
    desmap[desmap >1 ] = 1
    desmap *= mask
    desmap = norm_image(desmap) * 255
    
    desmap = desmap.astype(np.uint8)
    im_color = cv2.applyColorMap(desmap, cv2.COLORMAP_JET)
    return im_color

def cen_crop(input_image, dsize):
    h, w, c = input_image.shape
    offset_h = int((h - dsize[0])/2)
    offset_w = int((w - dsize[1])/2)
    
    input_image = input_image[offset_h:offset_h + dsize[0], offset_w:offset_w+dsize[1], :]
    return input_image

desmap_dir = "/Users/Geoff/Documents/my_git/data/desmap"
cam_list = file_io.get_dir_list(desmap_dir)

for cam_dir in cam_list:
    #cam_dir = "/Users/Geoff/Documents/my_git/data/desmap/253/"
    video_list = file_io.get_dir_list(cam_dir)
    for video in video_list:
        img_list = file_io.get_listfile(video, "jpg")
        mask_name = video+ "_msk_128.npy"
        mask = np.fromfile(mask_name, np.float32)
        mask = np.expand_dims(np.reshape(mask, (128, 128)), 2)
        mask = cen_crop(mask, (114,114))
        mask = np.squeeze(mask)
        #mask = np.tile(mask, (1,1,3))

        for img_name in img_list:
            img = cv2.imread(img_name)
            desmap_name = img_name.replace(".jpg", ".infer_desmap")
            desmap = opencv_plot(desmap_name, mask)

            img = cen_crop(img, (114,114))
            combine_img = np.hstack((img, desmap))
            save_name = img_name.replace(".jpg", "_combine.png")
            cv2.imwrite(save_name, combine_img)
            #cv2.imshow("combine", combine_img)
            #cv2.waitKey(0)


