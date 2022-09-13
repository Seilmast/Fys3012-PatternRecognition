import numpy as np 
import matplotlib.pyplot as plt 



mu1 = np.array([0,0])
mu2 = np.array([1,1])
sigma1 = np.sqrt(0.2)
sigma2 = np.sqrt(0.2)

#Generate datapoints for class 1 and 2
x_1 = np.random.multivariate_normal(mu1, np.array([[sigma1**2, 0], [0, sigma1**2]]), size=50)
x_2 = np.random.multivariate_normal(mu2, np.array([[sigma1**2, 0], [0, sigma1**2]]), size=50)

#Processing
# x_1 = np.delete(x_1, np.where(np.sum(x_1, axis=1) > 1), axis=0)
# x_2 = np.delete(x_2, np.where(np.sum(x_2, axis=1) < 1), axis=0)

x_1 = np.concatenate((x_1, np.ones((len(x_1), 1))), axis=1)
x_2 = np.concatenate((x_2, np.ones((len(x_2), 1))), axis=1)

#Combine data from each cluster
data = np.concatenate((x_1, x_2), axis=0)
labels = np.concatenate((np.ones(len(x_1))*-1,  #w_1 = -1
                         np.ones(len(x_2))),    #w_2 = 1
                         axis=0)

#One line to find the Sum of Error Squares Estimation
w = np.linalg.inv(data.T@data)@data.T@labels    #Eq. 3.45

x_axis = np.linspace(-1, 2, num=1000)
y_axis = -(w[0]*x_axis + w[2])/(w[1])

plt.figure()
plt.scatter(x_1[:,0], x_1[:,1], c='blue')
plt.scatter(x_2[:,0], x_2[:,1], c='red')
plt.plot(x_axis, y_axis, c='k')
plt.show()
