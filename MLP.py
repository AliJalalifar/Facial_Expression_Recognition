from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import cv2
import numpy
import glob
import os
import dlib
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC

emotions = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]

landmarks = []
fds = []
labels = []


for emotion in emotions:
    for feat_path in glob.glob(os.path.join("Parts\\dataset\\%s\\*.kp" % emotion)):
        #print feat_path
        fd = joblib.load(feat_path)
        fds.append(fd.flatten())
        labels.append(emotions.index(emotion))



pca = PCA(n_components=80,svd_solver='randomized',
          whiten=True)
pca.fit(fds)
joblib.dump(pca, "model/pca.model")
Xnew = pca.transform(fds)

clf = MLPClassifier(solver='adam', alpha=1e-5,activation='tanh',hidden_layer_sizes=(200,), random_state=0,verbose=True,max_iter=200,warm_start=True)
clf.fit(Xnew, labels)

# If feature directories don't exist, create them
if not os.path.isdir(os.path.split("model/MLP.model")[0]):
    os.makedirs(os.path.split("model/MLP.model")[0])
# Saving classifier
joblib.dump(clf, "model/MLP.model")