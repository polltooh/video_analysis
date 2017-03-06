import tensorflow as tf
from PIL import Image
import random
import numpy as np

def msc_data_arg(file_list, scale, img_w, img_h):
  img_file = tf.read_file(file_list[0])
  img = tf.image.decode_jpeg(img_file)

  desmap_file = tf.read_file(file_list[1])
  desmap = tf.decode_raw(desmap_file, tf.float32)
  desmap = tf.reshape(desmap, [img_h, img_w, 1])

  mask_file = tf.read_file(file_list[2])
  mask = tf.decode_raw(mask_file, tf.float32)
  mask = tf.reshape(mask, [img_h, img_w, 1])

  # resize image
  scale_data_w = int(scale * img_w)
  scale_data_h = int(scale * img_h)
  scale_img = tf.image.resize_images(img, [scale_data_h, scale_data_w],
                                     method=tf.image.ResizeMethod.BILINEAR)

  scale_desmap = tf.image.resize_images(desmap, [scale_data_h, scale_data_w],
                                        method=tf.image.ResizeMethod.BILINEAR)
  scale_desmap /= (scale**2)
  scale_mask = tf.image.resize_images(mask, [scale_data_h, scale_data_w],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return scale_img, scale_desmap, scale_mask


if __name__ == '__main__':
  file_list = [
    '../data/000094_128.jpg',
    '../data/000094_128.desmap',
    '../data/253-20160506-18_msk_128.npy'
  ]

  # Raw size: 227 * 227, min size: 170, crop size: ~160
  scales = [0.75, 1.0, 1.25, 1.5]
  scale = random.choice(scales)

  sess = tf.InteractiveSession()
  scale_img, scale_desmap, scale_mask = msc_data_arg(file_list, scale,
                                                     128, 128)

  img = Image.fromarray(np.round(scale_img.eval()).astype(np.uint8))
  img.save("../data/img.jpg")

  mask = Image.fromarray(np.squeeze(scale_mask.eval()), "1")
  mask.save("../data/mask.png")

  sess.close()

