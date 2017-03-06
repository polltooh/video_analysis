from TensorflowToolbox.utility import file_io
import shutil

file_list = file_io.read_file("ucsd_file_label_num.txt")
num_list = list()

for i, f in enumerate(file_list):
    if i != 0 and i % 200 == 0 and i != 2000:
        i = i-1
        shutil.copy2(file_list[i-1], file_list[i])        
