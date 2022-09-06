from cProfile import label
import numpy as np 
import matplotlib.pyplot as plt 


h = [0.05, 0.2]
N = [32, 256, 5000]

x_true = np.linspace(-1, 3, 1000)
p = np.copy(x_true)
p[p>2] = 0
p[p<0] = 0
p[p>0] = 0.5 

plt.figure()
plt.plot(x_true, p, label='True dist')
plt.show()

def normal_dist(x , mean=0 , sd=1):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density
    
plt.figure()
for hi in h:
    for ni in N:
        samples = np.random.uniform(low=0, high=2, size=ni)[:, np.newaxis]
        

        x_grid = np.linspace(-1, 3, 100)
        p_hat = []

        for xi in x_grid:
            p = np.sum(normal_dist((np.ones((ni,1))*xi - samples)/hi))/(hi*ni)
            p_hat.append(p)
        
        
        plt.plot(x_grid, p_hat, label=f'N={ni}, h={hi}')
        plt.legend()
        plt.show()