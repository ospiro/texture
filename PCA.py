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
labels = np.zeros(len(files))
random.shuffle(files)
arr = []
for i,f in enumerate(files):
    arr.append(np.load('/home/spiro/AlexNet/npz/'+f)['arr_0'])

splitpoint = 4000
valnum = len(arr)-splitpoint
accs = np.zeros([51,2])
for K in range(5, 256, 5):
    features = np.zeros((len(files),256*K))
    for i in range(len(arr)):
        pca = PCA(n_components=K)
        features[i,:] = np.reshape(pca.fit_transform(arr[i]),(1,-1))
        labels[i] =get_label('/home/spiro/AlexNet/npz/' + f)
        np.save('/home/spiro/AlexNet/PCA_features/K_'+str(K)+ '_' + str(labels[i]),features)
    clf = SGDClassifier()
    clf.partial_fit(features[:splitpoint],labels[:splitpoint],classes=np.unique(labels))
    preds = clf.predict(features[splitpoint:,:])
    np.save('/home/spiro/AlexNet/PCA_preds/K_'+str(K)+ '_' + str(labels[i]),preds)
    accs[i,:] = np.array([K,np.sum(preds==labels[splitpoint:])/float(valnum)])
np.save('/home/spiro/AlexNet/PCA_accs',accs)

#splitpoint = 4000
#valnum = features.shape[0]-splitpoint
#print "ASDF", valnum
#clf = SGDClassifier()
##for i in range(features[:splitpoint].shape[0]):
#clf.partial_fit(features[:splitpoint],labels[:splitpoint],classes=np.unique(labels))
#preds = clf.predict(features[splitpoint:,:])
#np.save('fullpreds',preds)
#train_acc = np.sum(preds==labels[splitpoint:])/float(valnum)
#print train_acc
