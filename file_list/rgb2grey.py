import cv2
import numpy as np

i = cv2.imread("Lily1.jpg")
i_grep = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
i_grep = cv2.resize(i_grep, (256, 256))
i_grep = i_grep.astype(np.float32)
i_grep.tofile("Lily_grep.npy")
print(i_grep)
