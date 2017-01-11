import cv2
import numpy as np

i = cv2.imread("traffic.jpg")
i_grep = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
i_grep = cv2.resize(i_grep, (256, 256))
i_grep = i_grep.astype(np.float32)
i_grep.tofile("traffic_gray.npy")
print(i_grep)
