from TensorflowToolbox.utility import file_io
import shutil

file_list = file_io.read_file("ucsd_file_label_num.txt")
num_list = list()

for i, f in enumerate(file_list):
    if i != 0 and i % 200 == 0 and i != 2000:
        i = i-1
        f_pre = file_list[i-1].split(" ")[0]
        f_curr = file_list[i].split(" ")[0]
        shutil.copy2(f_pre, f_curr)        
