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
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


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

X_train, X_test, y_train, y_test = train_test_split(fds, labels, random_state=0,test_size=0.4)

pca = PCA(n_components=75,svd_solver='randomized',
          whiten=True)
pca.fit(X_train)
X_train2 = pca.transform(X_train)

clf = MLPClassifier(solver='adam', alpha=1e-5,activation='tanh',hidden_layer_sizes=(200,), random_state=0,verbose=True,max_iter=200,warm_start=True)
clf.fit(X_train2, y_train)

Xtest2 = pca.transform(X_test)

y_pred = clf.predict(Xtest2)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = ((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=emotions,
                      title='Confusion matrix, without normalization',normalize=True)



plt.show()