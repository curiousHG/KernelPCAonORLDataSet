import os
import cv2
import numpy as np
from sklearn.utils import shuffle


def get_dataset(img_folder=None):
	if img_folder is None:
		img_folder = '../data/ORL_faces'
	img_data_array = []
	class_name = []
	for dir1 in os.listdir(img_folder):
		for file in os.listdir(os.path.join(img_folder, dir1)):
			image_path = os.path.join(img_folder, dir1, file)
			image = cv2.imread(image_path, 0)
			IMG_HEIGHT, IMG_WIDTH = (68, 56)
			image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
			image = np.array(image)
			image = image.ravel()
			image = image.astype('float32')
			# image /= 255
			img_data_array.append(image)
			class_name.append(dir1)
	img_data_array, class_name = shuffle(img_data_array, class_name, random_state=0)
	return img_data_array, class_name
