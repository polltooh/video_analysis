from TensorflowToolbox.utility import file_io
import sys
import numpy as np


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_list list_name.txt")
        exit(1)

    file_name = sys.argv[1]
    file_list = file_io.read_file(file_name)

    file_name_list = list()
    label_list = list()
    infer_list = list()

    file_dict = dict()
    label_dict = dict()

    for f in file_list:
        ff, ll, ii = f.split(" ")
        label_dict[ff] = ll
        if ff not in file_dict:
            file_dict[ff] = list()
            file_dict[ff].append(float(ii))
        else:
            file_dict[ff].append(float(ii))

    file_list = list()    
    for key in file_dict:
        curr_line = key + " " + label_dict[key] + " " + \
                    "%.2f"%np.mean(np.array(file_dict[key]))

        file_list.append(curr_line)

    file_list.sort()
    file_io.save_file(file_list, file_name)

