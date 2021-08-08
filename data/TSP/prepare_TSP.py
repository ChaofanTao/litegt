import numpy as np
import torch
import pickle
import time
import os
import matplotlib.pyplot as plt

# if not os.path.isfile('TSP.zip'):
#     print('downloading..')
#     !curl https://www.dropbox.com/s/1wf6zn5nq7qjg0e/TSP.zip?dl=1 -o TSP.zip -J -L -k
#     !unzip TSP.zip -d ../
#     # !tar -xvf TSP.zip -C ../
# else:
#     print('File already downloaded')

import os
os.chdir('../../') # go to root folder of the project
print(os.getcwd())

import pickle



from torch.utils.data import DataLoader

from data.data import LoadData
from data.TSP import TSP, TSPDatasetDGL, TSPDataset

start = time.time()

DATASET_NAME = 'TSP'
dataset = TSPDatasetDGL(DATASET_NAME) 

print('Time (sec):',time.time() - start) # ~ 30 mins

start = time.time()

with open('data/TSP/TSP.pkl','wb') as f:
    pickle.dump([dataset.train,dataset.val,dataset.test],f)
        
print('Time (sec):',time.time() - start) # 58s