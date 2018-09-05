from TensorflowToolbox.data_flow.data_ph_abs import DataPhAbs
import tensorflow as tf


class DataPh(DataPhAbs):
    def __init__(self, model_params):
        self.input_ph = tf.placeholder(tf.float32, shape=[
            # model_params["batch_size"],
            None,
            model_params["feature_ph_row"],
            model_params["feature_ph_col"],
            model_params["feature_cha"]],
                                       name="feature"
                                       )

        self.label_ph = tf.placeholder(tf.float32, shape=[
            # model_params["batch_size"],
            None,
            model_params["label_ph_row"],
            model_params["label_ph_col"],
            model_params["label_cha"]],
                                       name="label"
                                       )

        self.mask_ph = tf.placeholder(tf.float32, shape=[
            # model_params["batch_size"],
            None,
            model_params["mask_ph_row"],
            model_params["mask_ph_col"],
            model_params["mask_cha"]],
                                      name="mask"
                                      )

        # Add da loss weight
        self.da_label_ph = tf.placeholder(tf.float32, shape=[
            # model_params["batch_size"],
            None,
            model_params["label_ph_row"],
            model_params["label_ph_col"],
            model_params["label_cha"]],
                                          name="da_label"
        )

        self.da_loss_weight = tf.placeholder(tf.float32, shape=[
            1
        ],
                                             name="da_loss_weight")

    def get_da_weight(self):
        return self.da_loss_weight

    def get_da_label(self):
        return self.da_label_ph

    def get_label(self):
        return self.label_ph

    def get_input(self):
        return self.input_ph

    def get_mask(self):
        return self.mask_ph
