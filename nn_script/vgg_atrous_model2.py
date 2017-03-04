from TensorflowToolbox.model_flow.model_abs import ModelAbs
from TensorflowToolbox.model_flow import model_func as mf
from TensorflowToolbox.utility import image_utility_func as iuf

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
            conv11, [3, 3, 64, 64], [1, 1],
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
            "SAME", wd, "atrous5_1"), leaky_param)

        atrous52 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            atrous51, [3, 3, 512, 512], 2,
            "SAME", wd, "atrous5_2"), leaky_param)

        print(atrous52)
        #hyper_list.append(atrous52)

        hyper_list.append(atrous52)

        hypercolumn = self._pack_tensor_list(hyper_list)
        print(hypercolumn)

        [b, w, h, c]= hypercolumn.get_shape().as_list()
        conv6 = mf.add_leaky_relu(mf.convolution_2d_layer(
            hypercolumn, [1, 1, c, 512], [1, 1],
            "SAME", wd, "conv6"), leaky_param)

        deconv1 = self._deconv2_wrapper(conv6, conv21, 
                    256, wd, "deconv1")
        print(deconv1)

        deconv2 = self._deconv2_wrapper(deconv1, conv11, 
                    64, wd, "deconv2")
        print(deconv2)

        # Add domain classifier
        if model_params['use_da']:
            da_conv1 = mf.add_leaky_relu(mf.convolution_2d_layer(
                    deconv2, [1, 1, 64, 64], [1, 1],
                    "SAME", wd, "da_conv1"), leaky_param)
            da_cls = mf.add_leaky_relu(mf.convolution_2d_layer(
                    da_conv1, [1, 1, 64, 2], [1, 1],
                    "SAME", wd, "da_cls"), leaky_param)
            self.da_cls = da_cls

        #deconv1 = mf.deconvolution_2d_layer(conv6, [3, 3, 256, 512], 
        #            [2, 2], [b, 111, 111, 256], 'VALID', wd, 'deconv1')
        #print(deconv1)

        #deconv2 = mf.deconvolution_2d_layer(deconv1, [3, 3, 64, 256], 
        #            [2, 2], [b, 224, 224, 64], 'VALID', wd, 'deconv2')

        #print(deconv2)

        conv7 = mf.add_leaky_relu(mf.convolution_2d_layer(
            deconv2, [1, 1, 64, 1], [1, 1],
            "SAME", wd, "conv7"), leaky_param)
        print(conv7)

        #tf.add_to_collection("image_to_write", data_ph.get_input())
        #tf.add_to_collection("image_to_write", data_ph.get_label())
        #tf.add_to_collection("image_to_write", data_ph.get_mask()) 
        #tf.add_to_collection("image_to_write", conv7) 
        with tf.variable_scope("image_sum"):
            self._add_image_sum(data_ph.get_input(), data_ph.get_label(), 
                    conv7, data_ph.get_mask())

        self.predict_list = list()
        self.predict_list.append(conv7)

    def _add_image_sum(self, input_img, label, conv, mask):
        concat = iuf.merge_image(2, [input_img, label, conv, mask])
        tf.add_to_collection("image_to_write", concat)


    def _deconv2_wrapper(self, input_tensor, sample_tensor, 
                output_channel, wd, layer_name):
        [b, h, w, _] = sample_tensor.get_shape().as_list()
        [_,_,_,c] = input_tensor.get_shape().as_list()

        deconv = mf.deconvolution_2d_layer(input_tensor, 
                    [3, 3, output_channel, c], 
                    [2, 2], [b, h, w, output_channel], 'VALID', 
                    wd, layer_name)
        return deconv


    def _pack_tensor_list(self, tensor_list):
        hypercolumn = tf.concat(3, tensor_list)

        return hypercolumn

    def _resize_deconv(self, input_tensor, desire_shape,
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

    def _filter_mask(self, tensor, mask):
        tensor = tensor * mask
        return tensor

    def model_loss(self, data_ph, model_params):
        with tf.variable_scope("loss"):
            label = data_ph.get_label()
            mask = data_ph.get_mask()

            l2_loss_list = list()
            for i, deconv in enumerate(self.predict_list):
                deconv = self._filter_mask(deconv, mask)
                label = self._filter_mask(label, mask)
                l2_loss = mf.image_l2_loss(deconv, label, "l2_loss_%d" % i)
                l2_loss_list.append(l2_loss)
                tf.add_to_collection("losses", l2_loss)

                l1_loss = mf.image_l1_loss(deconv, label, "l1_loss_%d" % i)
                tf.add_to_collection("losses", l1_loss)

                count_diff = mf.count_diff(deconv, label, "count_diff")

                count = tf.reduce_sum(deconv, [1,2,3])
                label_count = tf.reduce_sum(label, [1,2,3])

            self.l1_loss = l1_loss
            self.count_diff = count_diff
            self.count = count
            self.label_count = label_count
            self.l2_loss = tf.add_n(l2_loss_list)

            # Add domain loss
            if model_params['use_da']:
                pred = tf.reshape(self.da_cls, [-1, 2])
                da_label = data_ph.get_da_label()
                total_da_loss = tf.nn.softmax_cross_entropy_with_logits(
                    pred, da_label
                )
                weight_da_loss = data_ph.get_da_weight() * total_da_loss
                self.da_loss = tf.reduce_mean(weight_da_loss)
                tf.add_to_collection("losses", self.da_loss)

            self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    def model_mini(self, model_params):
        with tf.variable_scope("optimization"):
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
    
    def get_l1_loss(self):
        return self.l1_loss

    def get_count_diff(self):
        return self.count_diff

    def get_count(self):
        return self.count

    def get_label_count(self):
        return self.label_count


if __name__ == "__main__":
    pass
