from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle as pkl

X = np.load("data.npy",allow_pickle=True)
y = np.load("target.npy",allow_pickle=True)
print(X.shape,y.shape)


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X,y)

pkl.dump(model,open("model.pkl","wb"))