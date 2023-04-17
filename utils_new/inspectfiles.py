import numpy as np
import os
import matplotlib.pyplot as plt
import _pickle as cPickle

path = '/home/ho11laqe/PycharmProjects/nnUNet_data/nnUNet_preprocessed/Task505_Glacier_mtl_boundary/nnUNetData_plans_mtl_2D_stage0/'
liste = os.listdir(path)
print(liste)
file = np.load(path + liste[0])
with open(path+liste[1], "rb") as pkl_file:
    pkl = cPickle.load(pkl_file)
    print()

data = file['data']
plt.imshow(data[0][0])
plt.show()
plt.imshow(data[0][0])
plt.show()
print()