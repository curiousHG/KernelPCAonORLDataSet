import numpy as np

import os
from PIL import Image

# for i in range(1,41):
# 	print(os.listdir(f'data/ORL_faces/s{i}'))
path = '../data/ORL_faces/s2'
images = os.listdir(f'{path}')
# imageArrays = []
sizeOfSingleImage = (112,92)
sum = np.zeros(sizeOfSingleImage,np.float64)
N = len(images)
for image in images:
	imarr = np.array(Image.open(f'{path}/{image}'),dtype=np.float64)
	sum += imarr/N

print(sum)
sum = np.array(np.round(sum), dtype=np.uint8)
out = Image.fromarray(sum,mode="L")
out.show()

