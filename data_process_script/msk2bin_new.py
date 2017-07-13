# coding=utf-8 
import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

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
    cv2.imwrite(pic_name, img)

root = os.getcwd()
files = os.listdir(root)
for file in files:
    if ".msk" in file:
        pic_name = file.replace(".msk", ".png")
        f = open(file, "r")
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
        print curve
        draw_curve(pic_name, curve)
        
os.system("pause")
        