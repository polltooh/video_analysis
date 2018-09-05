from TensorflowToolbox.model_flow import model_func
import tensorflow as tf

if __name__ == "__main__":
	img = tf.constant(3, dtype = tf.float32, shape = [1, 224, 224, 1])
	conv1 = model_func.convolution_2d_layer(img, \
				[3, 3, 1, 1], [2,2], "VALID", 0.01, "conv1")
    print(conv1)
	conv2 = model_func.convolution_2d_layer(conv1, \
				[3, 3, 1, 1], [2,2], "VALID", 0.01, "conv2")
	print(conv2)



