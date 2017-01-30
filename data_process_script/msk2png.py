# coding=utf-8 
import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from TensorflowToolbox.utility import file_io

data_dir = "../data"



def get_closed_curve(points):
    first = points[0]
    last = points[len(points)-1]
    if first[0] == 0:
        points.insert(0, [0,239])
    if last[0] == 351:
        points.append([351,239])
    return points
 
def draw_curve(pic_name, points):
	img = np.zeros((240,352,3), np.int32)
	pts = np.array(points, np.int32)
	img = cv2.fillPoly(img, [pts], (255,255,255))
	#cv2.imshow("image", img / 255)
	#cv2.waitKey(0)
	cv2.imwrite(pic_name, img)


cam_list = file_io.get_dir_list(data_dir)
for cam in cam_list:
	print(cam)
	msk_list = file_io.get_listfile(cam, "msk")
	for msk in msk_list:
		pic_name = msk.replace(".msk", ".png")
		f = open(msk, "r")
		lines = f.readlines()
		f.close()
		points = []
		for line in lines:
			if "[" in line and "," in line and "]" in line:
				a = line.replace("[","").replace("]","").replace("\n","").replace("\r","").split(",")
				x = int(a[0])
				y = int(a[1])
				if x <= 2:
					x = 0
				if x >= 349:
					x = 351
				if y <= 2:
					y = 0
				if y >= 237:
					y = 239
				points.append([x,y])
		curve = get_closed_curve(points)    
		draw_curve(pic_name, curve)

