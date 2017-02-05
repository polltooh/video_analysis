from TensorflowToolbox.model_flow.model_abs import ModelAbs
from TensorflowToolbox.model_flow import model_func as mf
import tensorflow as tf


# from traffic_data_ph import data_ph

class Model(ModelAbs):
    def __init__(self, data_ph, model_params):
        self.model_infer(data_ph, model_params)
        self.model_loss(data_ph, model_params)
        self.model_mini(model_params)

    def model_infer(self, data_ph, model_params):
        input_ph = data_ph.get_input()
        leaky_param = model_params["leaky_param"]
        wd = model_params["weight_decay"]

        hyper_list = list()

        print(input_ph)

        conv11 = mf.add_leaky_relu(mf.convolution_2d_layer(
            input_ph, [3, 3, 3, 64], [1, 1],
            "SAME", wd, "conv1_1"), leaky_param)

        atrous1 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            conv11, [3, 3, 64, 64], 2,
            "SAME", wd, "atrous1"), leaky_param)

        print(atrous1)
        hyper_list.append(atrous1)

        conv21 = mf.add_leaky_relu(mf.convolution_2d_layer(
            atrous1, [3, 3, 64, 128], [1, 1],
            "SAME", wd, "conv2_1"), leaky_param)

        atrous2 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            conv21, [3, 3, 128, 128], 2,
            "SAME", wd, "atrous2"), leaky_param)

        print(atrous2)
        hyper_list.append(atrous2)

        conv31 = mf.add_leaky_relu(mf.convolution_2d_layer(
            atrous2, [3, 3, 128, 256], [1, 1],
            "SAME", wd, "conv3_1"), leaky_param)

        conv32 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv31, [3, 3, 256, 256], [1, 1],
            "SAME", wd, "conv3_2"), leaky_param)

        atrous3 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            conv32, [3, 3, 256, 256], 2,
            "SAME", wd, "atrous3"), leaky_param)

        print(atrous3)
        hyper_list.append(atrous3)

        conv41 = mf.add_leaky_relu(mf.convolution_2d_layer(
            atrous3, [3, 3, 256, 512], [1, 1],
            "SAME", wd, "conv4_1"), leaky_param)

        conv42 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv41, [3, 3, 512, 512], [1, 1],
            "SAME", wd, "conv4_2"), leaky_param)
        atrous4 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            conv42, [3, 3, 512, 512], 2,
            "SAME", wd, "atrous4"), leaky_param)

        print(atrous4)
        hyper_list.append(atrous4)

        conv51 = mf.add_leaky_relu(mf.convolution_2d_layer(
            atrous4, [3, 3, 512, 512], [1, 1],
            "SAME", wd, "conv5_1"), leaky_param)

        conv52 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv51, [3, 3, 512, 512], [1, 1],
            "SAME", wd, "conv5_2"), leaky_param)

        atrous5 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            conv52, [3, 3, 512, 512], 2,
            "SAME", wd, "atrous5"), leaky_param)

        print(atrous5)

        hyper_list.append(atrous5)

        hypercolumn = self.pack_tensor_list(hyper_list)

        c_dimension = hypercolumn.get_shape().as_list()[3]

        conv6 = mf.add_leaky_relu(mf.convolution_2d_layer(
            hypercolumn, [1, 1, c_dimension, 1], [1, 1],
            "SAME", wd, "conv6"), leaky_param)

        self.predict_list = list()
        self.predict_list.append(conv6)

    def pack_tensor_list(self, tensor_list):
        hypercolumn = tf.concat(3, tensor_list)

        return hypercolumn

    def resize_deconv(self, input_tensor, desire_shape,
                      output_channel, wd, layer_name):
        b, w, h, c = input_tensor.get_shape().as_list()
        with tf.variable_scope(layer_name):
            if w != desire_shape[0] or h != desire_shape[1]:
                input_tensor = tf.image.resize_images(input_tensor,
                                                      [desire_shape[0],
                                                       desire_shape[1]])

            deconv = mf.deconvolution_2d_layer(input_tensor,
                                               [3, 3, output_channel, c],
                                               [1, 1],
                                               [b, desire_shape[0],
                                                desire_shape[1],
                                                output_channel],
                                               "SAME", wd, 'deconv')

        return deconv

    def filter_mask(self, tensor, mask):
        tensor = tensor * mask
        return tensor

    def model_loss(self, data_ph, model_params):
        label = data_ph.get_label()
        mask = data_ph.get_mask()

        l2_loss_list = list()
        for i, deconv in enumerate(self.predict_list):
            deconv = self.filter_mask(deconv, mask)
            label = self.filter_mask(label, mask)
            l2_loss = mf.l2_loss(deconv, label, "MEAN", "l2_loss_%d" % i)
            l2_loss_list.append(l2_loss)
            tf.add_to_collection("losses", l2_loss)

        self.l2_loss = tf.add_n(l2_loss_list)
        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    def model_mini(self, model_params):
        optimizer = tf.train.AdamOptimizer(
            model_params["init_learning_rate"],
            epsilon=1.0)
        self.train_op = optimizer.minimize(self.loss)

    def get_train_op(self):
        return self.train_op

    def get_loss(self):
        return self.loss

    def get_l2_loss(self):
        return self.l2_loss


if __name__ == "__main__":
    pass
