import numpy as np
import matplotlib.pyplot as plt

#print(np.linspace(10, 400, 20))

#print(np.logspace(1, np.log10(400), 20))

x1 = np.linspace(10, 400, 20)
x2 = np.logspace(np.log10(10), np.log10(400), 20)

y = np.zeros(20)

plt.figure()
plt.plot(x1, y, 'o')
plt.plot(x2, y+0.5, 'o')
plt.ylim([-0.5, 1])

period_want = np.logspace(np.log10(10), np.log10(400), 20)
rp_want = np.logspace(np.log10(4), np.log10(10), 20)

period_want2d, rp_want2d = np.meshgrid(period_want, rp_want)
plt.figure()
plt.plot(period_want2d, rp_want2d, marker='.', color='k', linestyle='none')




plt.show()