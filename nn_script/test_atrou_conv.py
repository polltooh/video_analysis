from TensorflowToolbox.model_flow import model_func
import tensorflow as tf

if __name__ == "__main__":
	img = tf.constant(3, dtype = tf.float32, shape = [1, 227,227, 3])
	atrous1 = model_func.atrous_convolution_layer(img, \
				[3,3, 3, 512], 4, "SAME", 0.01, "atrous1")
	print(atrous1)



