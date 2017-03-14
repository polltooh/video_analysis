from TensorflowToolbox.utility import file_io
import shutil
import os

def name_to_file_list(file_name):
    file_list = file_io.read_file(file_name)
    infer_list = list()
    mask_list = list()

    for i, f in enumerate(file_list):
        f = f.split(" ")[0]
        infer_name = f.replace(".jpg", ".infer_desmap")
        if not os.path.exists(infer_name):
            continue
        mask_name = "/".join(f.split("/")[:-1]) + "_msk_128.npy"

        infer_list.append(infer_name)
        mask_list.append(mask_name)
        file_list[i] = f

    return file_list, infer_list, mask_list

def copyfile(file_list, src_path, des_path):
    for f in file_list:
        new_file_name = f.replace(src_path, des_path)
        file_path = os.path.dirname(new_file_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        shutil.copy(f, new_file_name)

src_path = "/media/dog/data/WebCamT_60000"
des_path = "/media/dog/data/WebCamT_60000_desmap"

downtown_test_file_name = \
        "/home/shanghang/video_lstm/file_list/downtown_test_list1.txt"
file_list, infer_list, mask_list = name_to_file_list(downtown_test_file_name)
copyfile(file_list, src_path, des_path)
copyfile(infer_list, src_path, des_path)
copyfile(mask_list, src_path, des_path)

parkway_test_file_name = \
        "/home/shanghang/video_lstm/file_list/parkway_test_list1.txt"
file_list, infer_list, mask_list  = name_to_file_list(parkway_test_file_name)
copyfile(file_list, src_path, des_path)
copyfile(infer_list, src_path, des_path)
copyfile(mask_list, src_path, des_path)
