from TensorflowToolbox.utility import file_io
from TensorflowToolbox.utility import image_utility_func
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import xmltodict
from PIL import Image, ImageDraw
import sys

#dsize = (256, 256)

debug = False


def get_point(one_line):
    one_line = one_line[1:-2]
    p = one_line.split(",")
    p = np.array([float(p_s) for p_s in p], np.float32)
    return p


def load_mask(mask_file_name):
    mask_img = cv2.imread(mask_file_name, 0)
    bbox = image_utility_func.get_bbox(mask_img, 127)
    return bbox
    # p_l = list()
    # with open(mask_file_name, "r") as f:
    #    text = f.read()
    #    l = text.split("\n")
    #    for i in xrange(len(l) - 1):
    #        if (len(l[i+1]) == 0 or len(l[i+1]) == 1):
    #            break
    #        p_l.append(get_point(l[i + 1]))
    # p_l = np.array(p_l)
    # return p_l


    # mask_pt = np.loadtxt(mask_file_name)
    # print(mask_pt)


def get_density_map(annot_name, mask_coord):
    assert (annot_name.endswith(".xml"))
    img = np.zeros(dsize, np.float32)
    with open(annot_name) as xml_d:
        ss = xml_d.read()
        try:
            doc = xmltodict.parse(ss)
        except:
            try:
                ss = ss.replace("&", "")
                doc = xmltodict.parse(ss)
            except:
                print(annot_name + " cannot be read")
                return img

    def add_to_image(image, bbox, coord):
        xmax = int(bbox['xmax'])
        xmin = int(bbox['xmin'])
        ymax = int(bbox['ymax'])
        ymin = int(bbox['ymin'])

        xmax = int((xmax - coord[1]) * dsize[0] / float(coord[0] - coord[1]))
        xmin = int((xmin - coord[1]) * dsize[0] / float(coord[0] - coord[1]))
        ymax = int((ymax - coord[3]) * dsize[0] / float(coord[2] - coord[3]))
        ymin = int((ymin - coord[3]) * dsize[0] / float(coord[2] - coord[3]))

        density = 1 / float((ymax - ymin + 1) * (xmax - xmin + 1))
        image[ymin:ymax, xmin:xmax] += density
        return image

    if 'vehicle' not in doc['annotation']:
        print(annot_name + " no vehicle")
        return img

    for vehicle in doc['annotation']['vehicle']:
        if debug:
            vehicle = doc['annotation']['vehicle']
        bbox = vehicle['bndbox']
        add_to_image(img, bbox, mask_coord)

    return img


def crop_image(image_name, mask_bbox, save_data=False):
    image = cv2.imread(image_name)
    if image.shape[0] != 240:
        image = cv2.resize(image, (352, 240))
        print(image_name + " shape is " + str(image.shape))

    # for i in xrange(3):
    #    image[:,:,i] = image[:,:,i] * mask_bin

    # x_max = int(np.max(mask_pts[:,0]))
    # x_min = int(np.min(mask_pts[:,0]))
    # y_max = int(np.max(mask_pts[:,1]))
    # y_min = int(np.min(mask_pts[:,1]))

    x_min = mask_bbox[0]
    x_max = mask_bbox[0] + mask_bbox[2]
    y_min = mask_bbox[1]
    y_max = mask_bbox[1] + mask_bbox[3]

    mask_coord = np.array([x_max, x_min, y_max, y_min])

    croped_image = image[y_min:y_max, x_min:x_max]
    resize_image = cv2.resize(croped_image, dsize)

    annot_name = image_name.replace(".jpg", ".xml")
    if not os.path.exists(annot_name):
        print(annot_name + " is not exist. " + image_name + " is removed")
        os.remove(image_name)
        return

    density_map = get_density_map(annot_name, mask_coord)
    # cv2.imshow("test", density_map * 255)
    # cv2.waitKey(0)
    if save_data:
        save_image_name = image_name.replace(".jpg", "") + "_%d.jpg"%(dsize[0])
        cv2.imwrite(save_image_name, resize_image)
        density_map.tofile(save_image_name.replace(".jpg", ".desmap"))
        # cv2.imshow("resized", resize_image)
        # display_density_map = np.repeat(np.expand_dims(density_map,2) * 255, 3, axis = 2)
        # cv2.imshow("masked", display_density_map)
        # cv2.waitKey(0)


def gen_mask_image(mask_pts):
    img = Image.new("L", (352, 240), 0)
    ImageDraw.Draw(img).polygon(mask_pts, outline=1, fill=1)
    mask = np.array(img)
    return mask


if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Usage: msk_resize.py data_dir size_len")
        exit(1)

    data_dir = sys.argv[1]
    size_len = int(sys.argv[2])
    global dsize
    dsize = (size_len, size_len)

    mask_dir_list = file_io.get_dir_list(data_dir)

    for mask_dir in mask_dir_list:
        mask_list = file_io.get_listfile(mask_dir, ".msk")
        for mask in mask_list:
            image_dir_name = mask.replace(".msk", "")
            image_list = file_io.get_listfile(image_dir_name, "jpg")
            mask_name = mask.replace(".msk", ".png")
            mask_bbox = load_mask(mask_name)
            # if mask == "../data/data_new/Training_Data/Cam181/01.msk":
            #    mask_bin = cv2.imread(mask_dir + "/01_mask.jpg")
            #    mask_bin = mask_bin[:,:,1]
            #    mask_bin /= 255
            # else:
            #    try:
            #        mask_pts = load_mask(mask)
            #        mask_bin = gen_mask_image(mask_pts)
            #    except:
            #        mask_pts = load_mask(mask)
            #        print(mask)
            #        print("mask is wrong")
            #        exit(1)
            #        continue

            for img in image_list:
                #if img.endswith("_resize.jpg"):
                #    continue
                if len(img.split("_")) > 1:
                    continue
                try:
                    # crop_image(img, mask_bin, mask_pts, True)
                    crop_image(img, mask_bbox, True)
                except:
                    print(img)
                    debug = True
                    crop_image(img, mask_bbox, True)
                    debug = False
                    pass
