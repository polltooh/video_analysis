import tensorflow as tf
import TensorflowToolbox
from TensorflowToolbox.data_flow.data_input_abs import DataInputAbs


class DataInput(DataInputAbs):
    def __init__(self, filename):
        self.data_filenamequeue = tf.train.string_input_producer([filename],
                                                                 shuffle=False)
        self.load_data()

    def get_label(self):
        return self.label

    def get_input(self):
        return self.input

    def load_data(self):
        data_type_list = [tf.constant([], tf.string),
                          tf.constant([], tf.string)]
        line_reader = tf.TextLineReader()
        key, next_line = line_reader.read(self.data_filenamequeue)
        data_list = tf.decode_csv(next_line, data_type_list, field_delim=" ")
        self.input = data_list[0]
        self.label = data_list[1]


if __name__ == "__main__":
    test_input = DataInput("../file_list/test_file.txt")
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for i in range(10):
        label_v, input_v = sess.run([test_input.get_label(),
                                     test_input.get_input()])
        print(label_v)
        print(input_v)

    coord.request_stop()
    coord.join(threads)
