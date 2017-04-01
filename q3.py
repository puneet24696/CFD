import numpy as np
import matplotlib.pyplot as plt
import pylab
b = np.linspace(-4e-8, 4e-8, num=1000)
print b
#print(len(b))
c = []
for i in range(1000) :
    c.append((1-np.cos(b[i]))/(b[i]*b[i]))
#print c

c = np.array(c)
plt.figure(1)
plt.plot(b,c)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('sampleplot.png')
plt.show()

d=[]
for i in range(1000):
    d.append((np.sin(b[i]))**2/((b[i]**2)*(1+np.cos(b[i]))))
plt.figure(2)
plt.plot(b,d)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('sampleplotc.png')
plt.show()


