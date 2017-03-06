from TensorflowToolbox.utility import file_io
import sys
import numpy as np

def mae(label_list, infer_list):
    mae_v = np.mean(np.abs(label_list - infer_list))
    return mae_v

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analize_result.py file_name")
        exit(1)

    file_name = sys.argv[1]
    file_list = file_io.read_file(file_name)
    label_list = list()
    infer_list = list()

    for f in file_list:
        n, label, infer = f.split(" ")
        label_list.append(float(label)) 
        infer_list.append(float(infer))

    mae_v = mae(np.array(label_list), np.array(infer_list))
    print(mae_v)
