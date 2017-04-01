import numpy as np
import matplotlib.pylab as plt
x=[]
y=[]
z=[]
#for first order approx. forward difference
for i in range(50):
    x.append(np.pi*0.25*0.5**i)
    y.append(abs(((np.sin(np.pi*0.25 + x[i])-np.sin(np.pi*0.25))/x[i]-np.cos(np.pi*0.25))/np.cos(np.pi*0.25)))
    z.append((np.sin(np.pi*0.25 + x[i])-np.sin(np.pi*0.25))/x[i])
#print(x,y)
print(z)

x11=np.log(x)
y11=np.log(y)
plt.figure(1)
plt.plot(x11,y11)
plt.xlabel('logh')
plt.ylabel('logy')

plt.savefig('sampleplotq4.png')
plt.show()

#for second order approx. backwawrd difference
x2=[]
y2=[]
z2=[]
for i in range(50):
    x2.append(np.pi*0.25*0.5**i)
    y2.append(abs(((3*np.sin(np.pi*0.25)-4*np.sin(np.pi*0.25-x2[i])+np.sin(np.pi*0.25-2*x2[i]))/(2*x2[i])-np.cos(np.pi*0.25))/np.cos(np.pi*0.25)))
    z2.append((3*np.sin(np.pi*0.25)-4*np.sin(np.pi*0.25-x2[i])+np.sin(np.pi*0.25-2*x2[i]))/(2*x2[i]))
import matplotlib.pylab as plt2
x22=np.log(x2)
y22=np.log(y2)
plt.figure(2)
plt2.plot(x22,y22)
plt2.xlabel('logh')
plt2.ylabel('logy')

plt2.savefig('sampleplotq41.png')
plt2.show()
print(z2)



#for fourth  order approx. backwawrd difference
x3=[]
y3=[]
z3=[]
for i in range(50):
    x3.append(np.pi*0.25*0.5**i)
    y3.append(abs(((-np.sin(np.pi*0.25+2*x2[i])+8*np.sin(np.pi*0.25+x2[i])-8*np.sin(np.pi*0.25-x2[i])+np.sin(np.pi*0.25-2*x2[i]))/(12*x2[i])-np.cos(np.pi*0.25))/np.cos(np.pi*0.25)))
    z3.append((-np.sin(np.pi*0.25+2*x2[i])+8*np.sin(np.pi*0.25+x2[i])-8*np.sin(np.pi*0.25-x2[i])+np.sin(np.pi*0.25-2*x2[i]))/(12*x2[i]))

import matplotlib.pylab as plt3
x33=np.log(x3)
y33=np.log(y3)
plt.figure(3)
plt3.plot(x33,y33)
plt3.xlabel('logh')
plt3.ylabel('logy')

plt3.savefig('sampleplotq42.png')
plt3.show()
print(z3)
