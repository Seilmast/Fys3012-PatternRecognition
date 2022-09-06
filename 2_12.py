import numpy as np 
import matplotlib.pyplot as plt 


class normdist:
    def __init__(self, mu, sigma) -> None:
        '''
        mu: Can be a single value, or a vector
        sigma: single value
        '''
        
        self.mu = mu
        self.sigma = sigma
        
    def get_prob(self, x):
        p = np.exp(-((np.dot((x-self.mu),(x-self.mu))))/(2*self.sigma**2))/(2*np.pi*(self.sigma)**2)
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
        L = np.array([[0, 1], 
                      [0.5, 0]]) #Hardcoded just cause
                
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


mu1 = np.array([1,1])
mu2 = np.array([3,3])
sigma1 = np.sqrt(0.2)
sigma2 = np.sqrt(0.2)


dist1 = normdist(mu1, sigma1)
dist2 = normdist(mu2, sigma2)


#Generate datapoints for class 1 and 2
x_1 = np.random.multivariate_normal(mu1, np.array([[sigma1**2, 0], [0, sigma1**2]]), size=5000)
x_2 = np.random.multivariate_normal(mu2, np.array([[sigma1**2, 0], [0, sigma1**2]]), size=5000)

task = classifier(x_1, x_2, dist1, dist2)
true_class, pred_class = task.minimize_error()
conf_mat = task.create_conf_mat(true_class, pred_class)

print(np.sum(np.diag(conf_mat))/np.sum(conf_mat))

true_class, pred_class = task.minimize_risk()
conf_mat = task.create_conf_mat(true_class, pred_class)

print(np.sum(np.diag(conf_mat))/np.sum(conf_mat))

plt.figure()
plt.scatter(x_1[:,0], x_1[:,1], c='blue')
plt.scatter(x_2[:,0], x_2[:,1], c='red')
plt.show()
