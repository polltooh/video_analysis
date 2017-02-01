from traffic_data_ph import DataPh
from traffic_data_input import DataInput
#from vgg_model import Model
from vgg_atrous_model import Model
import tensorflow as tf
import cv2

class NetFlow(object):
	def __init__(self, model_params, load_train, load_test):
		self.load_train = load_train
		self.load_test = load_test
		self.model_params = model_params
		if load_train:
			self.train_data_input = DataInput(model_params, is_train = True)
		if load_test:
			self.test_data_input = DataInput(model_params, is_train = False)

		self.data_ph = DataPh(model_params)
		self.model = Model(self.data_ph, model_params)
		self.loss = self.model.get_loss()
		self.l2_loss = self.model.get_l2_loss()
		self.train_op = self.model.get_train_op()

	def get_feed_dict(self, sess, is_train):
		feed_dict = dict()
		if is_train:
			input_v, label_v ,mask_v, file_line_v = sess.run([
									self.train_data_input.get_input(), 
									self.train_data_input.get_label(),
									self.train_data_input.get_mask(),
									self.train_data_input.get_file_line()])
		else:
			input_v, label_v , mask_v, file_line_v = sess.run([
									self.test_data_input.get_input(), 
									self.test_data_input.get_label(),
									self.test_data_input.get_mask(),
									self.test_data_input.get_file_line()])

		feed_dict[self.data_ph.get_input()] = input_v
		feed_dict[self.data_ph.get_label()] = label_v
		feed_dict[self.data_ph.get_mask()] = mask_v

		return feed_dict

	def check_feed_dict(self, feed_dict):
		data_list = list()

		for key in feed_dict:
			data_list.append(feed_dict[key])

		cv2.imshow("image", data_list[0][0])	
		cv2.imshow("label", data_list[1][0] * 255)	
		cv2.imshow("mask", data_list[2][0])	
		cv2.waitKey(0)
		
		
	def mainloop(self):
		sess = tf.Session()
		init_op = tf.initialize_all_variables()
		sess.run(init_op)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord, sess = sess)
		if self.load_train:
			for i in range(self.model_params["max_training_iter"]):
				feed_dict = self.get_feed_dict(sess, is_train = True)
				self.check_feed_dict(feed_dict)

				_, loss_v = sess.run([self.train_op, self.loss], feed_dict)		
				print("train loss: %.4f "%loss_v)

				if i % self.model_params["test_per_iter"] == 0:
					feed_dict = self.get_feed_dict(sess, is_train = False)
					l2_loss_v = sess.run(self.l2_loss, feed_dict)		
					print("test loss: %.4f"%l2_loss_v)
		else:
			for i in range(self.model_params["test_iter"]):
				feed_dict = self.get_feed_dict(sess, is_train = False)
				loss_v = sess.run(self.loss, feed_dict)		
				print(loss_v)
			

		coord.request_stop()
		coord.join(threads)
