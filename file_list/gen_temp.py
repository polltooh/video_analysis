import numpy as np

a = np.zeros((256, 256), np.float32)
a.tofile("temp1.npy")

b = np.ones((256, 256), np.float32)
b.tofile("temp2.npy")
