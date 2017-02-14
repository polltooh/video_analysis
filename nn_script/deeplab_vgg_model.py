import tensorflow as tf
from six.moves import cPickle
from TensorflowToolbox.model_flow import model_func as mf

# Loading net skeleton with parameters name and shapes.
with open("../models/net_skeleton.ckpt", "rb") as f:
    net_skeleton = cPickle.load(f)

"""
# The DeepLab-LargeFOV model can be represented as follows:
## input -> [conv-relu](dilation=1, channels=64) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=128) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=256) x 3 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=512) x 3 -> [max_pool](stride=1)
##       -> [conv-relu](dilation=2, channels=512) x 3 -> [max_pool](stride=1) -> [avg_pool](stride=1)
##       -> [conv-relu](dilation=12, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=21) -> [pixel-wise softmax loss].
"""
num_layers = [2, 2, 3, 3, 3, 1, 1, 1]
dilations = [[1, 1],
             [1, 1],
             [1, 1, 1],
             [1, 1, 1],
             [2, 2, 2],
             [12],
             [1],
             [1]]
n_classes = 21
# All convolutional and pooling operations are applied using kernels of size 3x3
# padding is added so that the output of the same size as the input.
ks = 3


def create_variable(name, shape):
    """Create a convolution filter variable of the given name and shape,
       and initialise it using Xavier initialisation
       (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).
    """
    initialiser = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable


def create_bias_variable(name, shape):
    """Create a bias variable of the given name and shape,
       and initialise it to zero.
    """
    initialiser = tf.constant_initializer(value=0.0, dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable


class DeepLabLFOVModel(object):
    """DeepLab-LargeFOV model with atrous convolution and bilinear upsampling.

    This class implements a multi-layer convolutional neural network for semantic image segmentation task.
    This is the same as the model described in this paper: https://arxiv.org/abs/1412.7062 - please look
    there for details.
    """

    def __init__(self, model_params, data_ph, weights_path=None):
        """Create the model.

        Args:
          weights_path: the path to the cpkt file with dictionary of weights from .caffemodel.
        """
        self.data_ph = data_ph
        input_ph = self.data_ph.get_input()

        self.model_params = model_params
        self.variables = self._create_variables(weights_path)
        self.raw_output = self._create_network(input_ph,
                                               keep_prob=tf.constant(0.5))
        # Generate regression loss and train op
        self.model_mini()
        self.reg_loss()

    @staticmethod
    def _create_variables(weights_path):
        """Create all variables used by the network.
        This allows to share them between multiple calls
        to the loss function.

        Args:
          weights_path: the path to the ckpt file with dictionary of weights from .caffemodel.
                        If none, initialise all variables randomly.

        Returns:
          A dictionary with all variables.
        """
        var = list()

        if weights_path is not None:
            with open(weights_path, "rb") as f:
                weights = cPickle.load(f)  # Load pre-trained weights.
                for name, shape in net_skeleton:
                    var.append(tf.Variable(weights[name],
                                           name=name))
                del weights
        else:
            # Initialise all weights randomly with the Xavier scheme,
            # and
            # all biases to 0's.
            for name, shape in net_skeleton:
                if "/w" in name:  # Weight filter.
                    w = create_variable(name, list(shape))
                    var.append(w)
                else:
                    b = create_bias_variable(name, list(shape))
                    var.append(b)
        return var

    def _create_network(self, input_ph, keep_prob):
        current = input_ph

        v_idx = 0  # Index variable.

        # Last block is the classification layer.
        for b_idx in xrange(len(dilations) - 1):
            for l_idx, dilation in enumerate(dilations[b_idx]):
                w = self.variables[v_idx * 2]
                b = self.variables[v_idx * 2 + 1]
                if dilation == 1:
                    conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1],
                                        padding='SAME')
                else:
                    conv = tf.nn.atrous_conv2d(current, w, dilation,
                                               padding='SAME')
                current = tf.nn.relu(tf.nn.bias_add(conv, b))
                v_idx += 1
            # Optional pooling and dropout after each block.
            if b_idx < 3:
                current = tf.nn.max_pool(current,
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
            elif b_idx == 3:
                current = tf.nn.max_pool(current,
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
            elif b_idx == 4:
                current = tf.nn.max_pool(current,
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                current = tf.nn.avg_pool(current,
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
            elif b_idx <= 6:
                current = tf.nn.dropout(current, keep_prob=keep_prob)

        # TODO: change to a regression layer, shape now (batch, w/8, h/8, 1024)
        # Up-sample
        current = tf.image.resize_bilinear(current,
                                           tf.shape(input_ph)[1:3])
        current = mf.convolution_2d_layer(current, [1, 1, 1024, 1], [1, 1],
                                          "SAME",
                                          wd=self.model_params["weight_decay"],
                                          layer_name="Conv2D_24")
        return current

    @staticmethod
    def filter_mask(tensor, mask):
        tensor = tensor * mask
        return tensor

    def reg_loss(self):
        label = self.data_ph.get_label()
        mask = self.data_ph.get_mask()

        raw_output = self.filter_mask(self.raw_output, mask)  # mask output
        label = self.filter_mask(label, mask)                 # mask label
        self.l2_loss = mf.l2_loss(raw_output, label, "MEAN", "l2_loss")

    # Add APIs
    def model_mini(self):
        optimizer = tf.train.AdamOptimizer(
            self.model_params["init_learning_rate"],
            epsilon=1.0)
        self.train_op = optimizer.minimize(self.l2_loss)

    def get_train_op(self):
        return self.train_op

    def get_l2_loss(self):
        return self.l2_loss

    def get_loss(self):
        return self.l2_loss

