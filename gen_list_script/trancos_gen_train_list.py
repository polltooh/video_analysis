from TensorflowToolbox.utility import file_io

trancos_data_path = "../"

def file_list_to_train_list(file_list_name):
    train_file = file_io.read_file(file_list_name)
    train_file = [trancos_data_path + t for t in train_file]
    train_file = [t + " " + t.replace(image_ext, desmap_ext) + " " \
                    + t.replace(image_ext, mask_ext) for t in train_file]

    return train_file 

if __name__ == "__main__":
    image_ext = ".jpg"
    desmap_ext = ".desmap"
    mask_ext = "_mask.npy"
    file_list_dir = "../file_list/"
    save_train_file_name = "trancos_train_list1.txt"
    save_test_file_name = "trancos_test_list1.txt"

    train_file = file_list_to_train_list("../file_list/trancos_org_trainval.txt")
    test_file = file_list_to_train_list("../file_list/trancos_org_test.txt")

    file_io.save_file(train_file, file_list_dir + save_train_file_name)
    file_io.save_file(train_file, file_list_dir + save_test_file_name)
