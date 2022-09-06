import numpy as np 
import matplotlib.pyplot as plt 

mu1 = np.array([1,1])
mu2 = np.array([1.5,1.5])
sigma1 = np.sqrt(0.2)
sigma2 = np.sqrt(0.2)

n = 50

#Samples from each class
class_1 = np.random.multivariate_normal(mu1, np.array([[sigma1**2, 0], [0, sigma1**2]]), size=n)
class_2 = np.random.multivariate_normal(mu2, np.array([[sigma1**2, 0], [0, sigma1**2]]), size=n)

x_1 = np.random.multivariate_normal(mu1, np.array([[sigma1**2, 0], [0, sigma1**2]]), size=n*2)
x_2 = np.random.multivariate_normal(mu2, np.array([[sigma1**2, 0], [0, sigma1**2]]), size=n*2)


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
        return conf_mat
        
        
## N = 1

NN = NNclassifier(class_1, class_2)
pred1, pred2 = NN.classify(x_1, 1), NN.classify(x_2, 1)
true1, true2 = np.zeros(len(x_1)), np.ones(len(x_2))

true_lab = np.concatenate((true1, true2), axis=0)
pred = np.concatenate((pred1, pred2), axis=0)
data = np.concatenate((x_1, x_2), axis=0)
NN.accuracy(pred, true_lab)

plt.figure()


for x, lab in zip(data, pred):
    if lab == 0:
        plt.scatter(x[0], x[1], c = 'red')
    else:
        plt.scatter(x[0], x[1], c = 'blue')
    
plt.scatter(class_1[:,0], class_1[:,1], c='red', marker='x')
plt.scatter(class_2[:,0], class_2[:,1], c='blue', marker='x')
plt.show()

## N = 3

pred1, pred2 = NN.classify(x_1, 3), NN.classify(x_2, 3)
true1, true2 = np.zeros(len(x_1)), np.ones(len(x_2))

true_lab = np.concatenate((true1, true2), axis=0)
pred = np.concatenate((pred1, pred2), axis=0)
data = np.concatenate((x_1, x_2), axis=0)
NN.accuracy(pred, true_lab)

plt.figure()


for x, lab in zip(data, pred):
    if lab == 0:
        plt.scatter(x[0], x[1], c = 'red')
    else:
        plt.scatter(x[0], x[1], c = 'blue')
    
plt.scatter(class_1[:,0], class_1[:,1], c='red', marker='x')
plt.scatter(class_2[:,0], class_2[:,1], c='blue', marker='x')
plt.show()
