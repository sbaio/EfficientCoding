## Wed 17 Feb 2017 21:01:12 PM CET 
## Author : SBAI Othman

# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob

# load the images
image_list = []
for filename in glob.glob('images/*.png'):
	im=cv2.imread(filename,0)
	image_list.append(im)


def find_max_rectangle(image):

	# First pass: for each pixel, calculate number of possible pixel below to make a rectangle
	down = np.empty_like(image,dtype=int)
	(row,col) = down.shape

	for i in reversed(range(row)):
		for j in range(col):
			if i==row-1:
				# if white pixel in last row
				if image[-1][j]:
					down[-1][j] = 1
				# black pixel in last row
				else:
					down[-1][j] = -1
			else:
				# white pixel
				if image[i][j]:
					#case previous pixel in the same column is white
					if down[i+1][j]>0:
						down[i][j] = down[i+1][j]+1
					# case previous pixel in the same column is black
					else:
						down[i][j] = 1
				# black pixel
				else:
					#print 'black pixel'
					down[i][j] = -1

	# Second pass: calculate the number of columns possible for the rectangle

	surf = np.empty_like(image,dtype=int)
	for i in range(row):
		for j in range(col):
			
			# if white pixel
			if image[i][j]:
				k=1
				while k+j<col and down[i][j+k]>=down[i][j]:
					k = k+1

				surf[i][j] = k*down[i][j]

			else:
				surf[i][j] = 0


	(i,j) = np.unravel_index(surf.argmax(), surf.shape)
	h = down[i,j]
	w = surf[i,j]/h

	return (j,i,j+w,i+h)


for index,image in enumerate(image_list):
	(xmin,ymin,xmax,ymax) = find_max_rectangle(image)
	clone  = image.copy()
	cv2.rectangle(image, (xmin, ymin), (xmax,ymax), (0, 255, 0), 2)
	cv2.imshow('result%d'%index,image)
