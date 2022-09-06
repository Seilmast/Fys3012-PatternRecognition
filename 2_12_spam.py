import numpy as np 
import scipy.io as io

class normdist:
    def __init__(self, mu, sigma) -> None:
        '''
        mu: Can be a single value, or a vector
        sigma: single value
        '''
        
        self.mu = mu
        self.sigma = sigma
        
    def get_prob(self, x):
        p = np.exp(-0.5*(x - self.mu).T @ self.sigma @ (x-self.mu))/(np.sqrt((np.pi*2)**(len(self.mu) * np.linalg.det(self.sigma))))
        return p 

class classifier:
    def __init__(self, x1, x2, dist1, dist2) -> None:
        self.x1 = x1
        self.x2 = x2 
        self.dist1 = dist1
        self.dist2 = dist2 
    
    
    def minimize_error(self):
        x = np.concatenate((self.x1, self.x2), axis=0)
        x1_len, x2_len = len(self.x1), len(self.x2)
        
        true_class = np.concatenate((np.zeros(x1_len), np.ones(x2_len)), axis=0)
        predicted_class = np.zeros((x1_len + x2_len))
        
        for i, vec in enumerate(x): 
            p0, p1 = self.dist1.get_prob(vec), self.dist2.get_prob(vec)
            if p1 > p0: 
                predicted_class[i] = 1
            
        return true_class, predicted_class 

    def minimize_risk(self):
        L = np.array([[0, 1], [0.5, 0]]) #Hardcoded just cause
                
        x = np.concatenate((self.x1, self.x2), axis=0)
        x1_len, x2_len = len(self.x1), len(self.x2)
        
        true_class = np.concatenate((np.zeros(x1_len), np.ones(x2_len)), axis=0)
        predicted_class = np.zeros((x1_len + x2_len))
        
        
        for i, vec in enumerate(x): 
            p0, p1 = self.dist1.get_prob(vec), self.dist2.get_prob(vec)

            if p0/p1 < L[1,0]/L[0,1]: 
                predicted_class[i] = 1
            
        return true_class, predicted_class 

    def create_conf_mat(self, true_class, pred_class):
        classes = len(np.unique(true_class))
        conf_mat = np.zeros((classes, classes))
        
        for y, y_hat in zip(true_class, pred_class):
            y, y_hat = int(y), int(y_hat)
            conf_mat[y, y_hat] += 1
        
        return conf_mat

xtr, xte = io.loadmat('data/Xtr_spam.mat')['Xtr_spam'].T, io.loadmat('data/Xte_spam.mat')['Xte_spam'].T
ytr, yte = io.loadmat('data/ytr_spam.mat')['ytr_spam'].flatten(), io.loadmat('data/yte_spam.mat')['yte_spam'].flatten()

mu1 = xtr[np.where(ytr==1), :][0].mean(axis=0)
mu2 = xtr[np.where(ytr==-1), :][0].mean(axis=0)


sigma1 = xtr[np.where(ytr==1), :].std(1)
sigma2 = xtr[np.where(ytr==-1), :].std(1)
sigma = (sigma1+sigma2)/2

S = np.eye(xtr.shape[1])*sigma

dist1 = normdist(mu2, S)# np.eye(xtr.shape[1])*sigma2)
dist2 = normdist(mu1, S)# np.eye(xtr.shape[1])*sigma1)

task = classifier(xtr[np.where(ytr==-1), :][0], xtr[np.where(ytr==1), :][0], dist1, dist2)
true_class, pred_class = task.minimize_error()
conf_mat = task.create_conf_mat(true_class, pred_class)

print(conf_mat)
print(np.sum(np.diag(conf_mat))/np.sum(conf_mat))

true_class, pred_class = task.minimize_risk()
conf_mat = task.create_conf_mat(true_class, pred_class)

print(np.sum(np.diag(conf_mat))/np.sum(conf_mat))
