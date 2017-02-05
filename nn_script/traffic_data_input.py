from TensorflowToolbox.data_flow.data_input_abs import DataInputAbs
from TensorflowToolbox.data_flow import data_reader
from TensorflowToolbox.data_flow import data_class
import tensorflow as tf
import cv2
import numpy as np


class DataInput(DataInputAbs):
    def __init__(self, model_params, is_train):
        arg_dict = self.get_arg_dict(model_params)
        self.is_train = is_train

        if is_train:
            file_name = model_params["train_file_name"]
        else:
            file_name = model_params["test_file_name"]

        self.file_queue = data_reader.file_queue(file_name, is_train)
        self.load_data(model_params, arg_dict)

    def get_arg_dict(self, model_params):
        arg_dict = dict()

        arg_dict["feature"] = dict()
        arg_dict["label"] = dict()
        arg_dict["mask"] = dict()

        for key in model_params:
            if "data_arg" in key:
                _, domain, field = key.split(".")
                arg_dict[domain][field] = model_params[key]
        arg_dict_list = list()
        arg_dict_list.append(arg_dict["feature"])
        arg_dict_list.append(arg_dict["label"])
        arg_dict_list.append(arg_dict["mask"])
        return arg_dict_list

    # if "label_arg" in key:
    #	_, field = key.split(".")
    #	label_arg_dict[field] = model_params[key]
    # arg_dict = [data_arg_dict, label_arg_dict]

    # return arg_dict_list

    def get_label(self):
        return self.label

    def get_input(self):
        return self.input

    def get_mask(self):
        return self.mask

    def get_file_line(self):
        return self.file_line

    def load_data(self, model_params, arg_dict):
        input_class = data_class.DataClass(tf.constant([], tf.string))
        input_class.decode_class = data_class.JPGClass(
            shape=[model_params["feature_row"],
                   model_params["feature_col"]],
            channels=model_params["feature_cha"])

        label_class = data_class.DataClass(tf.constant([], tf.string))
        label_class.decode_class = data_class.BINClass(
            [model_params["label_row"],
             model_params["label_col"],
             model_params["label_cha"]])

        mask_class = data_class.DataClass(tf.constant([], tf.string))
        mask_class.decode_class = data_class.BINClass(
            [model_params["mask_row"],
             model_params["mask_col"],
             model_params["mask_cha"]])

        tensor_list = [input_class] + [label_class] + [mask_class]
        batch_tensor_list = data_reader.file_queue_to_batch_data(
            self.file_queue,
            tensor_list, self.is_train,
            model_params["batch_size"],
            arg_dict)

        self.input = batch_tensor_list[0]
        self.label = batch_tensor_list[1]
        self.mask = batch_tensor_list[2]
        self.file_line = batch_tensor_list[3]


if __name__ == "__main__":
    """ example of running the code"""
    model_params = dict()
    model_params["feature_row"] = 256
    model_params["feature_col"] = 256
    model_params["feature_cha"] = 3

    model_params["label_row"] = 256
    model_params["label_col"] = 256
    model_params["label_cha"] = 1

    model_params["mask_row"] = 256
    model_params["mask_col"] = 256
    model_params["mask_cha"] = 1

    model_params["batch_size"] = 2

    model_params["data_arg.rflip_leftright"] = True
    model_params["data_arg.rflip_updown"] = False
    model_params["data_arg.rcrop_size"] = [200, 200]

    # model_params["label_arg.rflip_leftright"] = True
    # model_params["label_arg.rflip_updown"] = True
    # model_params["label_arg.rcrop_size"] = [200, 200, 1]


    model_params["train_file_name"] = "../file_list/test_file2.txt"

    # arg_dict_list = [data_arg_dict, label_arg_dict]

    train_input = DataInput(model_params, True)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for i in range(10):
        label_v, input_v, file_line_v = sess.run([train_input.get_label(),
                                                  train_input.get_input(),
                                                  train_input.get_file_line()])
        combined = np.hstack(
            (np.expand_dims(input_v[0][:, :, 1], axis=2), label_v[0] / 255))
        cv2.imshow("image", combined)
        cv2.waitKey(0)
    # print(file_line_v)

    coord.request_stop()
    coord.join(threads)
