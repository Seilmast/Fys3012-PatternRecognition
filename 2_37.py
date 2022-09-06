import numpy as np 
import matplotlib.pyplot as plt 

x = np.linspace(-1, 3, 1000)
p = np.copy(x)
p[p>2] = 0
p[p<0] = 0
p[p>0] = 0.5 

plt.figure()
plt.plot(x, p)
plt.show()


N = 5000
X = np.random.uniform(low=0, high=2, size=N)[:, np.newaxis]

k_values = [32, 64, 256]

for ki in k_values:
    x_grid = np.linspace(0, 2, 100)

    p_hat = []
    for xi in x_grid:
        idx = np.argsort(np.abs(X-xi)[:,0])
        nn = idx[:ki]
        xnn = X[nn,0]

        vx = np.max(xnn) - np.min(xnn)
        p_hat.append(p)

    plt.figure()
    plt.plot(x_grid, p_hat)
    plt.show()