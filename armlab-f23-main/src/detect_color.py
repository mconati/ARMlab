#!/usr/bin/env python3

# import the necessary packages
import numpy as np
import argparse
import cv2

yellow = [0, 255, 255]  # yellow in BGR colorspace
red = [0, 0, 255]
blue = [255, 0, 0]
green = [0, 255, 0]       # green
violet = [238, 130, 238]   # violet
orange = [255, 165, 0]     # orange

# # detect and calibrate color
# def get_limits(color):
#     c = np.uint8([[color]])  # BGR values
#     hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

#     hue = hsvC[0][0][0]  # Get the hue value
#     print('hue: ', hue)

#     # Handle red hue wrap-around
#     if hue >= 165:  # Upper limit for divided red hue
#         lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
#         upperLimit = np.array([180, 255, 255], dtype=np.uint8)
#     elif hue <= 15:  # Lower limit for divided red hue
#         lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
#         upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
#     else:
#         lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
#         upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
#     print('lowerlim:', lowerLimit)
#     print('upperlim:', upperLimit)
#     return lowerLimit, upperLimit


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
# load the image
image = cv2.imread(args["image"])
print('image shape: ', image.shape)

# # fetch color hue limits
# red_lower, red_upper = get_limits(color=red)
# yellow_lower, yellow_upper = get_limits(color=yellow)
# violet_lower, violet_upper = get_limits(color=violet)
# blue_lower, blue_upper = get_limits(color=blue)
# green_lower, green_upper = get_limits(color=green)
# orange_lower, orange_upper = get_limits(color=orange)

image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # 8 bit image

# define GBR (not RGB for some reason)
# defined in tuples, first tuple is the lower bound, second tuple is the upper bound
boundaries = [
	([20, 80, 60], [35, 255, 255]),   # yellow
	([120, 60, 60], [165, 255, 255]),    # violet
	([100, 120, 70], [128, 255, 255]),   # blue
	([55, 60, 40], [80, 255, 255]),    # green
	([8, 120, 70], [12, 255, 255])    # orange
]
red_boundaries = [
	([165, 30, 50], [180, 255, 255]),     # red upper
	([0, 30, 50], [6, 255, 255])     # red lower
]
# boundaries = [
# 	(red_lower, red_upper),     # red
# 	(yellow_lower, yellow_upper),   # yellow
# 	(violet_lower, violet_upper),    # violet
# 	(blue_lower, blue_upper),   # blue
# 	(green_lower, green_upper),    # green
# 	(orange_lower, orange_upper)    # orange
# ]

color_list = [
	'yellow',
	'violet',
	'blue',
	'green',
	'orange',
	'red'
]

# loop over the boundaries
for index, (lower, upper) in enumerate(boundaries):
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	# remove image noise
	kernel_size = 2
	kernel = np.ones((kernel_size,kernel_size), np.uint8)
	mask = cv2.erode(mask, kernel, iterations=2)
	mask = cv2.dilate(mask, kernel, iterations=2)
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for c in contours:
		area = cv2.contourArea(c)
		if area > 500 and area < 3000:
			x, y, w, h = cv2.boundingRect(c)
			cv2.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), 3)
			rect_area = w*h
			print(color_list[index], ' rect_area:', rect_area)
	# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
	output = cv2.bitwise_and(image, image, mask = mask)

	# show the images
	cv2.imshow("images", cv2.cvtColor(np.hstack([image, output]), cv2.COLOR_HSV2BGR))
	cv2.waitKey(0)

red_mask_list = []
for (lower, upper) in red_boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	# remove image noise
	kernel_size = 2
	kernel = np.ones((kernel_size,kernel_size), np.uint8)
	mask = cv2.erode(mask, kernel, iterations=2)
	mask = cv2.dilate(mask, kernel, iterations=2)
	#get all non zero values
	# coord = cv2.findNonZero(mask)
	# avg_coord = np.mean(coord, axis=0)
	# print(avg_coord)

	red_mask_list.append(mask)
	print('got here')

combine_red_mask = cv2.bitwise_or(red_mask_list[0], red_mask_list[1])
# further remove image noise after combining masks
kernel_size = 3
kernel = np.ones((kernel_size,kernel_size), np.uint8)
combine_red_mask = cv2.dilate(combine_red_mask, kernel, iterations=2)
combine_red_mask = cv2.erode(combine_red_mask, kernel, iterations=2)
output = cv2.bitwise_and(image, image, mask=combine_red_mask)
contours, hierarchy = cv2.findContours(combine_red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
	area = cv2.contourArea(c)
	if area > 500 and area < 3000:
		x, y, w, h = cv2.boundingRect(c)
		cv2.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), 3)
		rect_area = w*h
		print('red rect_area:', rect_area)

# show the images
cv2.imshow("images", cv2.cvtColor(np.hstack([image, output]), cv2.COLOR_HSV2BGR))
cv2.waitKey(0)

cv2.destroyAllWindows()
