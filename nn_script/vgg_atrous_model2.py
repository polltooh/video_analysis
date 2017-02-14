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

        conv12 = mf.add_leaky_relu(mf.convolution_2d_layer(
            input_ph, [3, 3, 3, 64], [1, 1],
            "SAME", wd, "conv1_2"), leaky_param)

        conv12_maxpool = mf.maxpool_2d_layer(conv12, [3, 3],
                                             [2, 2], "maxpool1")

        print(conv12_maxpool)
        #hyper_list.append(conv12_maxpool)

        conv21 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv12_maxpool, [3, 3, 64, 128], [1, 1],
            "SAME", wd, "conv2_1"), leaky_param)

        conv22 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv21, [3, 3, 128, 128], [1, 1],
            "SAME", wd, "conv2_2"), leaky_param)

        conv22_maxpool = mf.maxpool_2d_layer(conv22, [3, 3],
                                             [2, 2], "maxpool2")

        print(conv22_maxpool)
        hyper_list.append(conv22_maxpool)

        conv31 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv22_maxpool, [3, 3, 128, 256], [1, 1],
            "SAME", wd, "conv3_1"), leaky_param)

        conv32 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv31, [3, 3, 256, 256], [1, 1],
            "SAME", wd, "conv3_2"), leaky_param)

        atrous3 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            conv32, [3, 3, 256, 256], 2,
            "SAME", wd, "atrous3"), leaky_param)
        #conv33 = mf.add_leaky_relu(mf.convolution_2d_layer(
        #    conv32, [3, 3, 256, 256], [1, 1],
        #    "SAME", wd, "conv3_3"), leaky_param)

        #conv33_maxpool = mf.maxpool_2d_layer(conv33, [3, 3],
        #                                     [2, 2], "maxpool3")

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

        atrous51 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            atrous4, [3, 3, 512, 512], 2,
            "SAME", wd, "atrous51"), leaky_param)

        atrous52 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            atrous51, [3, 3, 512, 512], 2,
            "SAME", wd, "atrous52"), leaky_param)

        print(atrous52)
        #hyper_list.append(atrous52)

        hyper_list.append(atrous52)

        hypercolumn = self.pack_tensor_list(hyper_list)
        print(hypercolumn)

        [b, w, h, c]= hypercolumn.get_shape().as_list()
        conv6 = mf.add_leaky_relu(mf.convolution_2d_layer(
            hypercolumn, [1, 1, c, 512], [1, 1],
            "SAME", wd, "conv6"), leaky_param)

        deconv1 = mf.deconvolution_2d_layer(conv6, [3, 3, 256, 512], 
                    [2, 2], [b, 111, 111, 256], 'VALID', wd, 'deconv1')
        print(deconv1)

        deconv2 = mf.deconvolution_2d_layer(deconv1, [3, 3, 64, 256], 
                    [2, 2], [b, 224, 224, 64], 'VALID', wd, 'deconv2')

        print(deconv2)
        conv7 = mf.add_leaky_relu(mf.convolution_2d_layer(
            deconv2, [1, 1, 64, 1], [1, 1],
            "SAME", wd, "conv7"), leaky_param)
        print(conv7)

        tf.add_to_collection("image_to_write", data_ph.get_input())
        tf.add_to_collection("image_to_write", data_ph.get_label())
        tf.add_to_collection("image_to_write", data_ph.get_mask()) 
        tf.add_to_collection("image_to_write", conv7) 

        self.predict_list = list()
        self.predict_list.append(conv7)

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
            l2_loss = mf.image_l2_loss(deconv, label, "MEAN", "l2_loss_%d" % i)
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
