from sklearn import neighbors
import os 
from PIL import Image
import numpy as np
import shutil

x = []
y = []

for label in os.listdir('train'):
    for file in os.listdir('train/{}'.format(label)):
        im = Image.open('train/{}/{}'.format(label, file))
        im_gray = im.convert('L')
        pix =np.array(im_gray)
        pix = (pix > 150) * 1
        pix = pix.ravel()
        x.append(list(pix))
        y.append(int(label))
        train_x = np.array(x)
        train_y = np.array(y)
        model = neighbors.KNeighborsClassifier(n_neighbors=32)
        model.fit(train_x, train_y)

        


