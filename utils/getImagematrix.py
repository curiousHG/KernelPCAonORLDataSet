import numpy as np

import os
from PIL import Image

path = "../data/ORL_faces/"
subjects = os.listdir(path)


def createMatrix():
	for subject in subjects:
		subjectImages = os.listdir(f"{path}/{subject}")
		for image in subjectImages:
			with Image.open(f"{path}/{subject}/{image}") as im:
				x = np.ravel(im).T
				b = x.reshape(-1,1)

			break
		break


createMatrix()
