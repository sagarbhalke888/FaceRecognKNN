import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder 
from  tqdm import tqdm
from PIL import Image as im

path = r'E:\Desktop\Sunday ML\FaceRecognKNN\Data'


finalData = []
labels = []
for filename in tqdm(os.listdir(path)):
    file = im.open(os.path.join(path,filename))
    finalData.append(np.asarray(file,dtype = int).flatten())
    labels.append(filename.split("_")[0])

print(len(finalData),len(labels))

np.save("data",np.array(finalData))
np.save("target",np.array(labels))
