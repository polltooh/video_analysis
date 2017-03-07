from TensorflowToolbox.utility import file_io

def get_file_dict():
    file_dir = "/media/dog/data/WebCamT_60000/train_test_separation/" 

    parkway_train_file = file_dir + "Parkway_Train.txt"

    parkway_test_file = file_dir + "Parkway_Test.txt"

    downtown_train_file = file_dir + "Downtown_Train.txt"

    downtown_test_file = file_dir + "Downtown_Test.txt"

    parkway_train = set(file_io.read_file(parkway_train_file))
    parkway_test = set(file_io.read_file(parkway_test_file))
    downtown_train = set(file_io.read_file(downtown_train_file))
    downtown_test = set(file_io.read_file(downtown_test_file))

    return parkway_train, parkway_test, downtown_train, downtown_test

def save_file(file_list, file_name):
    file_io.save_file(file_list, file_list_dir + file_name, True)

def check_add_to_list(file_dict, file_list, file_name, full_name):
    if file_name in file_dict:
        file_list.append(full_name)
    return file_list

if __name__ == "__main__":
    train_list = file_io.read_file("../file_list/train_list2.txt")
    test_list = file_io.read_file("../file_list/test_list2.txt")
    full_list = train_list + test_list

    file_list_dir = "../file_list/"
    
    parkway_train, parkway_test, downtown_train, downtown_test = get_file_dict()

    parkway_train_list = list()
    parkway_test_list = list()
    downtown_train_list = list()
    downtown_test_list = list()

    for f in full_list:
        cam_dir_name = f.split(" ")[0].split("/")[-2]

        check_add_to_list(parkway_train, parkway_train_list, cam_dir_name, f)
        check_add_to_list(parkway_test, parkway_test_list, cam_dir_name, f)
        check_add_to_list(downtown_train, downtown_train_list, cam_dir_name, f)
        check_add_to_list(downtown_test, downtown_test_list, cam_dir_name, f)


    parkway_train_name = 'parkway_train_list1.txt'
    parkway_test_name = 'parkway_test_list1.txt'

    save_file(parkway_train_list, parkway_train_name)
    save_file(parkway_test_list, parkway_test_name)

    downtown_train_name = 'downtown_train_list1.txt'
    downtown_test_name = 'downtown_test_list1.txt'

    save_file(downtown_train_list, downtown_train_name)
    save_file(downtown_test_list, downtown_test_name)



