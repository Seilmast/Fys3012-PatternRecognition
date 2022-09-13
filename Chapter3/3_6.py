import numpy as np 
import matplotlib.pyplot as plt 



mu1 = np.array([0,0])
mu2 = np.array([1,1])
sigma1 = np.sqrt(0.2)
sigma2 = np.sqrt(0.2)

#Generate datapoints for class 1 and 2
x_1 = np.random.multivariate_normal(mu1, np.array([[sigma1**2, 0], [0, sigma1**2]]), size=50)
x_2 = np.random.multivariate_normal(mu2, np.array([[sigma1**2, 0], [0, sigma1**2]]), size=50)

# # Processing
# x_1 = np.delete(x_1, np.where(np.sum(x_1, axis=1) > 1), axis=0)
# x_2 = np.delete(x_2, np.where(np.sum(x_2, axis=1) < 1), axis=0)

x_1 = np.concatenate((x_1, np.ones((len(x_1), 1))), axis=1)
x_2 = np.concatenate((x_2, np.ones((len(x_2), 1))), axis=1)

#Combine data from each cluster
data = np.concatenate((x_1, x_2), axis=0)
labels = np.concatenate((np.ones(len(x_1))*-1,  #w_1 = -1
                         np.ones(len(x_2))),    #w_2 = 1
                         axis=0)

w = np.random.rand(1,3) #[x1, x2, bias]
w_old = None 
convergence = False 
p = 0.1
epoch = 0

while not convergence:
    w_old = np.copy(w)
    update = np.zeros(3)
    
    for x, y in zip(data, labels):

        mult = x@w.T                    #Figure out which side of the decision boundary point x is 
        y_hat = -1 if mult < 0 else 1   #classify
        update += p*x*(y - y_hat)       #update, Eq. 3.40

    w = w + update 
    
    if (w == w_old).all() or epoch > 10:
        convergence = True
    
    epoch += 1
    


x_axis = np.linspace(-1, 2, num=1000)
y_axis = -(w[0,0]*x_axis + w[0,2])/(w[0,1])

plt.figure()
plt.scatter(x_1[:,0], x_1[:,1], c='blue')
plt.scatter(x_2[:,0], x_2[:,1], c='red')
plt.plot(x_axis, y_axis, c='k')
plt.plot(x_axis, y_axis, c='k')

plt.show()
