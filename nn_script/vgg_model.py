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

		hyper_list = list()
		deconv_list = list()

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
		hyper_list.append(conv12_maxpool)
		
		conv21 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv12_maxpool, [3, 3, 64, 128], [1,1], 
				"SAME", wd, "conv2_1"), leaky_param)

		conv22 = mf.add_leaky_relu(mf.convolution_2d_layer(
				conv21, [3, 3, 128, 128], [1,1], 
				"SAME", wd, "conv2_2"), leaky_param)

		conv22_maxpool = mf.maxpool_2d_layer(conv22, [2,2], 
				[2,2], "maxpool2")

		print(conv22_maxpool)
		hyper_list.append(conv22_maxpool)

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
		hyper_list.append(conv33_maxpool)

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
		hyper_list.append(conv43_maxpool)

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

		conv53_maxpool = conv53_maxpool

		print(conv53_maxpool)

		hyper_list.append(conv53_maxpool)
		concat1= self.pack_tensor_list(hyper_list)#hypercolumn feature

		deconv1 = self.resize_deconv(concat1,
					[224, 224], 3, wd, 'deconv1')

		deconv_list.append(deconv1)

		concat2 = self.pack_tensor_list([concat1, deconv1])
		deconv2 = self.resize_deconv(concat2,
					[224,224], 3, wd, 'deconv2')

		deconv_list.append(deconv2)

		concat3 = self.pack_tensor_list([concat1, deconv2])
		deconv3 = self.resize_deconv(concat3,
					[224,224], 3, wd, 'deconv3')
		deconv_list.append(deconv3)
		self.deconv_list = deconv_list

		#deconv11 = mf.deconvolution_2d_layer(conv53_maxpool, 
		#		[3, 3, 512, 256], [1,1], [b, 56, 56, 256], 'SAME', wd, 'deconv11')
		#deconv11_relu = mf.add_leaky_relu(deconv11, leaky_param)

    	#deconv12 = mf.deconvolution_2d_layer(deconv11_relu, [3, 3, 256, 256], [1,1], [b, 56, 56, 256], 'SAME', wd, 'deconv12')
    	#deconv12_relu = mf.add_leaky_relu(deconv12, leaky_param)

    	#deconv21 = mf.deconvolution_2d_layer(deconv12_relu, [3, 3, 128, 256], [2,2], [b, 113, 113, 128], 'VALID', wd, 'deconv21')
    	#deconv21_relu = mf.add_leaky_relu(deconv21, leaky_param)
    	#deconv22 = mf.deconvolution_2d_layer(deconv21_relu, [3, 3, 128, 128], [1,1], [b, 113, 113, 128], 'SAME', wd, 'deconv22')
    	#deconv22_relu = mf.add_leaky_relu(deconv22, leaky_param)
    
    	#deconv31 = mf.deconvolution_2d_layer(deconv22_relu, [3, 3, 64, 128], [2,2], [b, 227, 227, 64], 'VALID', wd, 'deconv31')
    	#deconv31_relu = mf.add_leaky_relu(deconv31, leaky_param)
		#self.out = tf.image.resize_images(conv53_maxpool, 224, 224)

	def pack_tensor_list(self, tensor_list):
		num = len(tensor_list)
		max_h = 0;
		max_w = 0;
		for t in tensor_list:
			b, w, h, c = t.get_shape().as_list()
			max_h = max(h, max_h)
			max_w = max(w, max_w)

		hypercolumn = tf.image.resize_images(tensor_list[0], max_h, max_w)

		for i in range(num-1):
			resized_tensor = tf.image.resize_images(tensor_list[i+1],
							max_h, max_w)
			hypercolumn = tf.concat(3, [hypercolumn, resized_tensor])

		return hypercolumn
			

	def resize_deconv(self, input_tensor, desire_shape, 
				output_channel, wd, layer_name):
		b, w, h, c = input_tensor.get_shape().as_list()
		with tf.variable_scope(layer_name):
			if w != desire_shape[0] or h != desire_shape[1]:
				input_tensor = tf.image.resize_images(input_tensor, 
								desire_shape[0], desire_shape[1])

			deconv = mf.deconvolution_2d_layer(input_tensor, 
					[3, 3, output_channel, c],[1,1], 
					[b, desire_shape[0], desire_shape[1], output_channel],
					"SAME", wd, 'deconv')
			
		return deconv

		
	#def test_size(self):
	#	input_layer = tf.zeros([1,224,224,3], tf.float32)
	#	wd = 0.01

	#	conv = mf.convolution_2d_layer(input_layer, 
	#			[3, 3, 3, 3], [2,2], "VALID", wd, "conv")
	#	print(conv)
	#	exit(1)
	#	#deconv11 = mf.deconvolution_2d_layer(self.conv53_maxpool, 
	#	#		[3, 3, 256, 512], [10,10], [b, 56, 56, 256], 'SAME', wd, 'deconv11')
	#	#deconv11_relu = mf.add_leaky_relu(deconv11, leaky_param)
		
		
	def model_loss(self, data_ph, model_params):
		label = data_ph.get_label()
		#self.l2_loss = mf.l2_loss(self.out, label, "MEAN", "l2_loss")
		#tf.add_to_collection("losses", self.l2_loss)
		l2_loss_list = list()
		for i, deconv in enumerate(self.deconv_list):
			l2_loss = mf.l2_loss(deconv, label, "MEAN", "l2_loss_%d"%i)
			l2_loss_list.append(l2_loss)
			tf.add_to_collection("losses", l2_loss)

		self.l2_loss = tf.add_n(l2_loss_list)
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
