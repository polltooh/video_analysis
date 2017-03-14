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
    

    def _single_hydra_cnn(self, input_ph, model_params, stage):
        
        leaky_param = model_params["leaky_param"]
        wd = model_params["weight_decay"]
        batch_size = model_params["batch_size"]

        with tf.variable_scope("stage_%d"%stage):
            conv1 = mf.add_leaky_relu(mf.convolution_2d_layer(
                input_ph, [7, 7, 3, 32], [1, 1],
                "SAME", wd, "conv1"), leaky_param)

            conv1_maxpool = mf.maxpool_2d_layer(conv1, [2, 2],
                                                 [2, 2], "maxpool1")

            conv2 = mf.add_leaky_relu(mf.convolution_2d_layer(
                conv1_maxpool, [7, 7, 32, 32], [1, 1],
                "SAME", wd, "conv2"), leaky_param)

            conv2_maxpool = mf.maxpool_2d_layer(conv2, [2, 2],
                                                 [2, 2], "maxpool2")

            conv3 = mf.add_leaky_relu(mf.convolution_2d_layer(
                conv2_maxpool, [3, 3, 32, 32], [1, 1],
                "SAME", wd, "conv3"), leaky_param)

            conv4 = mf.add_leaky_relu(mf.convolution_2d_layer(
                conv3, [1, 1, 32, 1000], [1, 1],
                "SAME", wd, "conv4"), leaky_param)
            
            conv5 = mf.add_leaky_relu(mf.convolution_2d_layer(
                conv4, [1, 1, 1000, 400], [1, 1],
                "SAME", wd, "conv5"), leaky_param)

            reshape_fc = tf.reshape(conv5, [batch_size, -1])
       
        return reshape_fc
                
    def _center_crop(input_ph, frac):
        b, h, w, c =  input_ph.get_shape.as_list()
        new_h = int(frac * h)
        new_w = int(frac * h)
        
    def model_infer(self, data_ph, model_params):
        input_ph = data_ph.get_input()
        leaky_param = model_params["leaky_param"]
        wd = model_params["weight_decay"]

        image_ph_0 = tf.image.resize_images(input_ph, [72, 72])
        stage_0 = self._single_hydra_cnn(image_ph_0, model_params, 0)
        print(stage_0)

        image_ph_1 = iuf.batch_center_crop_frac(image_ph_0, 0.9)
        image_ph_1 = tf.image.resize_images(image_ph_1, [72, 72])
        stage_1 = self._single_hydra_cnn(image_ph_1, model_params, 1)
        print(stage_1)

        image_ph_2 = iuf.batch_center_crop_frac(image_ph_0, 0.8)
        image_ph_2 = tf.image.resize_images(image_ph_2, [72, 72])
        stage_2 = self._single_hydra_cnn(image_ph_2, model_params, 2)
        print(stage_2)

        concat_feature = tf.concat(1, [stage_0, stage_1, stage_2])
        print(concat_feature)

        fc6 = mf.fully_connected_layer(concat_feature, 1024, wd, "fc6")
        fc7 = mf.fully_connected_layer(fc6, 1024, wd, "fc7")
        fc8 = mf.fully_connected_layer(fc7, 1024, wd, "fc8")
        
        batch_size = model_params["batch_size"]
        output = tf.expand_dims(tf.reshape(fc8, [batch_size, 32, 32]), 3)
        print(output)

        l_ph_r = model_params["label_ph_row"]
        l_ph_c = model_params["label_ph_col"]

        resized_output = tf.image.resize_images(output, [l_ph_r, l_ph_c])
        print(resized_output)

        with tf.variable_scope("image_sum"):
            self._add_image_sum(data_ph.get_input(), data_ph.get_label(), 
                    resized_output, data_ph.get_mask())

        self.predict_list = list()
        self.predict_list.append(resized_output)

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
