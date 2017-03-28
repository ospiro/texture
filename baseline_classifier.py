import sklearn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from scipy.misc import imread,imresize
import random
from os import listdir
label_name_file = '/home/spiro/DTD_LABELS.txt'
def get_label(im_fn):
    with open(label_name_file,'r') as l:
        label_names = l.read().splitlines()
    label_dict = {}
    for i, lab in enumerate(label_names):
        label_dict.update({lab:i})
    arr_fn = im_fn.split('_')
    return label_dict[arr_fn[6]]
files = listdir('/home/spiro/AlexNet/npz')
features = np.zeros((len(files),256*256))
labels = np.zeros(len(files))
random.shuffle(files)
for i,f in enumerate(files):
    arr = np.load('/home/spiro/AlexNet/npz/'+f)
    features[i,:] = np.reshape(arr['arr_0'],(1,-1))
    labels[i] =get_label('/home/spiro/AlexNet/npz/' + f)
print features
print labels
print features.shape
print labels.shape


splitpoint = 4000
valnum = features.shape[0]-splitpoint
print "ASDF", valnum
clf = SGDClassifier()
#for i in range(features[:splitpoint].shape[0]):
clf.partial_fit(features[:splitpoint],labels[:splitpoint],classes=np.unique(labels))
preds = clf.predict(features[splitpoint:,:])
np.save('fullpreds',preds)
train_acc = np.sum(preds==labels[splitpoint:])/float(valnum)
print train_acc
