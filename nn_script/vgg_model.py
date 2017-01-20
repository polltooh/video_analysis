from TensorflowToolbox.model_flow.model_abs import ModelAbs
from TensorflowToolbox.model_flow import model_func as mf
import tensorflow as tf
#from traffic_data_ph import data_ph

class Model(ModelAbs):
	def __init__(self, data_ph, model_params):
		self.model_infer(data_ph, model_params)
		self.model_loss(data_ph, model_params)
		self.model_mini(model_params)

	def model_infer(self, data_ph, model_params):
		input_ph = data_ph.get_input()
		leaky_param = model_params["leaky_param"]
		wd = model_params["weight_decay"]

		print(input_ph)

		conv11 = mf.add_leaky_relu(mf.convolution_2d_layer(
				input_ph, [3, 3, 3, 64], [1,1], 
				"SAME", wd, "conv1_1"), leaky_param)

		conv12 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv11, [3, 3, 64, 64], [1,1], 
				"SAME", wd, "conv1_2"), leaky_param)

		conv12_maxpool = mf.maxpool_2d_layer(conv12, [2,2], 
				[2,2], "maxpool1")

		print(conv12_maxpool)

		conv21 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv12_maxpool, [3, 3, 64, 128], [1,1], 
				"SAME", wd, "conv2_1"), leaky_param)

		conv22 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv21, [3, 3, 128, 128], [1,1], 
				"SAME", wd, "conv2_2"), leaky_param)

		conv22_maxpool = mf.maxpool_2d_layer(conv22, [2,2], 
				[2,2], "maxpool2")

		print(conv22_maxpool)

		conv31 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv22_maxpool, [3, 3, 128, 256], [1,1], 
				"SAME", wd, "conv3_1"), leaky_param)

		conv32 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv31, [3, 3, 256, 256], [1,1], 
				"SAME", wd, "conv3_2"), leaky_param)

		conv33 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv32, [3, 3, 256, 256], [1,1], 
				"SAME", wd, "conv3_3"), leaky_param)

		conv33_maxpool = mf.maxpool_2d_layer(conv33, [2,2], 
				[2,2], "maxpool3")

		print(conv33_maxpool)

		conv41 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv33_maxpool, [3, 3, 256, 512], [1,1], 
				"SAME", wd, "conv4_1"), leaky_param)

		conv42 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv41, [3, 3, 512, 512], [1,1], 
				"SAME", wd, "conv4_2"), leaky_param)

		conv43 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv42, [3, 3, 512, 512], [1,1], 
				"SAME", wd, "conv4_3"), leaky_param)

		conv43_maxpool = mf.maxpool_2d_layer(conv43, [2,2], 
				[2,2], "maxpool4")

		print(conv43_maxpool)

		conv51 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv43_maxpool, [3, 3, 512, 512], [1,1], 
				"SAME", wd, "conv5_1"), leaky_param)

		conv52 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv51, [3, 3, 512, 512], [1,1], 
				"SAME", wd, "conv5_2"), leaky_param)

		conv53 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv52, [3, 3, 512, 512], [1,1], 
				"SAME", wd, "conv5_3"), leaky_param)

		conv53_maxpool = mf.maxpool_2d_layer(conv53, [2,2], 
				[2,2], "maxpool5")

		self.out = tf.image.resize_images(conv53_maxpool, 224, 224)

		
	def model_loss(self, data_ph, model_params):
		label = data_ph.get_label()
		self.l2_loss = mf.l2_loss(self.out, label, "MEAN", "l2_loss")
		tf.add_to_collection("losses", self.l2_loss)
		self.loss = tf.add_n(tf.get_collection('losses'), name = 'total_loss')
	
	def model_mini(self, model_params):
		optimizer = tf.train.AdamOptimizer(
					model_params["init_learning_rate"], 
					epsilon = 1.0)
		self.train_op = optimizer.minimize(self.loss)
	
	def get_train_op(self):
		return self.train_op	

	def get_loss(self):
		return self.loss

	def get_l2_loss(self):
		return self.l2_loss


if __name__ == "__main__":
	pass
