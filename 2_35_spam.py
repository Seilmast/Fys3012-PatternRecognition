import scipy.io as io
import numpy as np 
import matplotlib.pyplot as plt 

xtr, xte = io.loadmat('data/Xtr_spam.mat')['Xtr_spam'].T, io.loadmat('data/Xte_spam.mat')['Xte_spam'].T
ytr, yte = io.loadmat('data/ytr_spam.mat')['ytr_spam'].flatten(), io.loadmat('data/yte_spam.mat')['yte_spam'].flatten()

mu1 = xtr[np.where(ytr==1), :][0].mean(axis=0)
mu2 = xtr[np.where(ytr==-1), :][0].mean(axis=0)


sigma1 = xtr[np.where(ytr==1), :].std(1)
sigma2 = xtr[np.where(ytr==-1), :].std(1)
sigma = (sigma1+sigma2)/2

S = np.eye(xtr.shape[1])*sigma

class NNclassifier:
    def __init__(self, class1, class2) -> None:
        self.class1 = class1
        self.class2 = class2
        self.label1 = np.zeros(len(self.class1))
        self.label2 = np.ones(len(self.class2))
        
        self.data = np.concatenate((self.class1, self.class2), axis=0)
        self.label = np.concatenate((self.label1, self.label2), axis=0)
        
    def classify(self, data, N):
        
        predicted_labels = np.zeros(len(data))
        
        for i, x in enumerate(data): 
            x = x.reshape(1, len(x))            
            dist = np.linalg.norm(self.data - x, axis=1)
            
            dist = dist.reshape(1, len(dist))
            lab = self.label.reshape(1, len(self.label))
            
            dist = np.concatenate((dist, lab), axis=0).T
            dist = dist[np.argsort(dist[:,0])]

            dist = dist[:N]
            lab = dist[:,1] #Labels of the N closest points
            lab, count = np.unique(lab, return_counts=True)
            predicted_labels[i] = lab[np.argmax(count)]
            
        return predicted_labels
    
    def accuracy(self, pred_lab, true_lab):
        n_classes = len(np.unique(true_lab))
        conf_mat = np.zeros((n_classes, n_classes))
        
        for y, y_hat in zip(true_lab, pred_lab):
            conf_mat[int(y), int(y_hat)] += 1
            
        print(f'Accuracy: {np.sum(np.diag(conf_mat))/np.sum(conf_mat)}')
        print(conf_mat)
        return conf_mat

NN_net = NNclassifier(xtr[np.where(ytr==-1), :][0], xtr[np.where(ytr==1), :][0])
pred = NN_net.classify(xte, 3)
print(np.unique(yte))
yte[yte<0] = 0
NN_net.accuracy(pred, yte)