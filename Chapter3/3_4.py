import numpy as np 
import matplotlib.pyplot as plt 

x1 = np.array([[0,0,1], [0,1,1]])
x2 = np.array([[1,0,1], [1,1,1]])

p = 0.01
w = np.array([[0,0,1]])
w_old = np.copy(w)
epochs = 100

data = np.concatenate((x1, x2), axis=0)
label = np.concatenate((np.ones(len(x1))*-1, 
                        np.ones(len(x2))), 
                       axis=0)


for epoch in range(epochs):
    y_hat = (w@data.T).flatten()
    y_hat[y_hat > 0] = 1
    y_hat[y_hat < 0] = -1
    
    update = 0
    
    for x, y, y_h in zip(data, label, y_hat):
        
        mult = y*((w@x))
        if mult >= 0:
            update += y*x
            
    w_old = np.copy(w)
    w = w - p*update
    
    if (w_old == w).all():
        break
    
lin_x = np.linspace(0, 1, num=1000)
lin_y = -(w[0,0]*lin_x + w[0,2])/(w[0,1]) 

plt.figure()
plt.scatter(x1[:,0], x1[:,1], c='red', label='Class 1')
plt.scatter(x2[:,0], x2[:,1], c='blue', label='Class 2')
plt.plot(lin_x, lin_y, c='k', label='Decision boundary')
plt.show()