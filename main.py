import numpy as np

import os
from PIL import Image

# for i in range(1,41):
# 	print(os.listdir(f'data/ORL_faces/s{i}'))
path = 'data/ORL_faces/s1'
images = os.listdir(f'{path}')
imageArrays = []
sizeOfSingleImage = (92, 112)
for image in images:
	with Image.open(f'{path}/{image}') as im:
		arr = np.array(im)
		flat_arr = arr.ravel()
		imageArrays.append(flat_arr)
x = np.mean(imageArrays,axis=0)
im = Image.fromarray(x,'L')
im.show()


