from TensorflowToolbox.utility import file_io
import numpy as np

new_dsize = (256, 256)

if __name__ == "__main__":
    label_path = "/media/dog/data/WorldExpo/train_label/"
    mask_dir = "/media/dog/data/WorldExpo/train_mask/"
    desmap_dir = "/media/dog/data/WorldExpo/gtDensity/gtDensity/"
    img_dir = "/media/dog/data/WorldExpo/train_frame/"

    img_list = file_io.get_listfile(img_dir, ".jpg")
    scale_str = str(new_dsize[0])

    for img_name in img_list:
        if not img_name.endswith("_" + scale_str + ".jpg"):
            continue
        label_name = img_name.replace(img_dir, desmap_dir).replace(".jpg", ".npy")
        mask_name = mask_dir + img_name.split("/")[-1].split("_")[0] \
                    + "_" + scale_str + ".npy"
        print(img_name)
        print(label_name)
        print(mask_name)
        exit(1)
