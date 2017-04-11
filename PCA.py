import sklearn
import numpy as np
from sklearn.decomposition import PCA,TruncatedSVD
#from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from scipy.misc import imread,imresize
import random
random.seed(1337)
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
random.shuffle(files)
N=5000
labels = np.zeros(N)
features = np.zeros((N, 256*256))
for i,f in enumerate(files):
    if i < N:
        features[i,:] = np.reshape(imread('/home/spiro/AlexNet/npz/'+f),(1,-1))#np.sqrt(np.reshape(imread('/home/spiro/AlexNet/npz/'+f),(1,-1))/np.linalg.norm(np.reshape(imread('/home/spiro/AlexNet/npz/'+f),(1,-1))))
        labels[i] = get_label('/home/spiro/AlexNet/npz/' + f)
splitpoint = N-300
valnum = N-splitpoint
accs = np.array([0,0])#np.zeros([51,2])
print "OK"
for K in [10]:#[10,25,50,100,150,100,250]:#range(5, 256, 5):
    #pca_features = np.zeros((2500,256*K))
    pca = PCA(n_components=K)
    #features[i,:] = np.reshape(pca.fit_transform(arr[i]),(1,-1))
    pca_features = pca.fit_transform(features)
    #pca_features = features
    print pca_features.shape
    #np.save('/home/spiro/AlexNet/PCA_features/K_'+str(K)+ '_' + str(labels[i]),features)
    clf = SVC()
    #clf = KNN(n_neighbors = 10)
    for i in range(pca_features.shape[0]):
        pca_features[i,:] = np.sign(pca_features[i,:])*np.sqrt(np.abs(pca_features[i,:])/np.linalg.norm(pca_features[i,:]))
    clf.fit(X=pca_features[:splitpoint,:],y=labels[:splitpoint]) 
    preds = clf.predict(pca_features[splitpoint:,:])
    #np.save('/home/spiro/AlexNet/PCA_preds/K_'+str(K)+ '_' + str(labels[i]),preds)
    acc = np.array([K,np.sum(preds==labels[splitpoint:])/float(valnum)])
    print acc
    np.vstack([accs,acc])
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
