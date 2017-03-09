from TensorflowToolbox.utility import file_io
import numpy as np
import os

new_dsize = (256, 256)

if __name__ == "__main__":
    train_list = list()
    test_list = list()

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

        if i == 0:
            file_list = train_list
        else:
            file_list = test_list

        img_list = file_io.get_listfile(img_dir, ".jpg")
        scale_str = str(new_dsize[0])

        for img_name in img_list:
            if not img_name.endswith("_" + scale_str + ".jpg"):
                continue
            label_name = img_name.replace(img_dir, desmap_dir).replace(".jpg", ".npy")
            mask_name = mask_dir + img_name.split("/")[-1].split("-")[0] \
                        + "_mask_" + scale_str + ".npy"

            if not os.path.exists(label_name):
                print(label_name, "is not exist")
                exit(1)

            if not os.path.exists(mask_name):
                print(name_name, "is not exist")
                exit(1)

            file_list.append(" ".join([img_name, label_name, mask_name]))
    
    train_file_name = "../file_list/world_expo_train_list1.txt"
    test_file_name = "../file_list/world_expo_test_list1.txt"

    file_io.save_file(train_list, train_file_name, True)
    file_io.save_file(test_list, test_file_name, True)
