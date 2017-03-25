#! /usr/bin/env python
from TensorflowToolbox.utility import file_io
import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit(1)
    file_name = sys.argv[1]
    file_list = file_io.read_file(file_name)
    file_list.sort()
    file_io.save_file(file_list, file_name)
