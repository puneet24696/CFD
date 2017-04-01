from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def timestep(xmin, xmax, xgrid, tmin, tmax, a, cfl):
    dx = (xmax - xmin)/(xgrid-1)
    dt  = cfl*dx/a
    tgrid = 1 + (tmax - tmin)/dt
    return tgrid , dt , dx

def grid(xmin, xmax ,dx ,tmin , tmax , dt):
    x, t = np.mgrid[xmin: xmax: dx, tmin: tmax: dt]
    bc_grid = x
    return bc_grid
#print grid(10,10)

def bc1(bc_grid):
    bc_grid[:, :] = 0
    bc_grid[0, :] = 1
    bc_grid[1:, 0] = 0
    return bc_grid

def bc2_1(bc_grid, xgrid):
    bc_grid[1:,1:] = 0
    bc_grid[0, :] = 0
    for i in range(0, len(bc_grid[:,0])):
        bc_grid[i, 0] = np.sin(2*np.pi*bc_grid[i,0])
    return bc_grid

#def bc3():


def bc2_2(bc_grid, xgrid):
    bc_grid[1:, 1:] = 0
    bc_grid[0, :] = 0
    bc_grid[:,0] = np.sin(2*np.pi*bc_grid[:,0]) + np.sin(20*np.pi*bc_grid[:,0])
    return bc_grid

def plot_grid(bc_grid,xmin , xgrid, tgrid, title):
    for t in range(len(bc_grid[0,:])):
        plt.plot(np.linspace(xmin, 1,len(bc_grid[:,0])), bc_grid[:, t])
        print t , len(bc_grid[0,:])
        plt.title(title )
        plt.xlabel("x")
        plt.ylabel("u")
        plt.waitforbuttonpress(0.001)
        plt.cla()



def FTFS(bc_grid, xgrid, tgrid, cfl):
    #print int(tgrid)
    #print int(xgrid)
    #print len(bc_grid[0,:])
    #print len(bc_grid[:,0])

    for t in range(0, len(bc_grid[0,:])-1):
        for x in range(0, len(bc_grid[:,0])-2):
			bc_grid[x+1,t+1] = bc_grid[x+1,t] - cfl*(bc_grid[x+2,t] - bc_grid[x+1,t])

    for t in range(0, len(bc_grid[0,:])-1):
       	bc_grid[len(bc_grid[:,0])-1,t+1] = bc_grid[len(bc_grid[:,0])-1,t] + cfl*(bc_grid[len(bc_grid[:,0])-2,t] - bc_grid[len(bc_grid[:,0])-1,t])

    return bc_grid



def FTCS(bc_grid, xgrid, tgrid, cfl):
    for t in range(0, len(bc_grid[0,:])-1):
        for x in range(0, len(bc_grid[:,0])-2):
            bc_grid[x+1, t+1] = bc_grid[x+1, t] - cfl*(bc_grid[x+2, t] - bc_grid[x, t])/2

    for t in range(0, len(bc_grid[0,:])-1):
       	bc_grid[len(bc_grid[:,0])-1,t+1] = bc_grid[len(bc_grid[:,0])-1,t] + cfl*(bc_grid[len(bc_grid[:,0])-2,t] - bc_grid[len(bc_grid[:,0])-1,t])

    return bc_grid

def FTBS(bc_grid, xgrid, tgrid, cfl):
    for t in range(0,len(bc_grid[0,:])-1):
        for x in range(0, len(bc_grid[:,0])-2):
            bc_grid[x+1,t+1] = bc_grid[x+1,t] + cfl*(bc_grid[x,t] - bc_grid[x+1,t])

    return bc_grid

def FTCS2(bc_grid, xgrid, tgrid, cfl):
    for t in range(0, len(bc_grid[0,:])-1 ):
        for x in range(-1, len(bc_grid[:,0])-1):
            bc_grid[x+1, t+1] = bc_grid[x+1, t] - cfl*(bc_grid[(x+2)%len(bc_grid[:,0]), t] - bc_grid[x, t])/2 + (cfl**2)*( bc_grid[(x+2)%len(bc_grid[:,0]),t] - 2*bc_grid[x+1,t] +bc_grid[x,t])/2

    return bc_grid

def FTBS2(bc_grid, xgrid, tgrid, cfl):
    for t in range(0,len(bc_grid[0,:])-1):
        for x in range(-1, len(bc_grid[:,0])-1):
            bc_grid[x+1,t+1] = bc_grid[x+1,t] + cfl*(bc_grid[x,t] - bc_grid[x+1,t])

    return bc_grid



'''
#Question1
string = [FTFS, FTCS, FTBS]
string1 = ["FTFS","FTCS", "FTBS"]

xgrid = 50.0
CFL = [0.5 , 1.0 , 1.5]
for i in range(3):
    for j in range(3):
        tgrid,dt,dx = timestep(0.0, 1.0, xgrid, 0.0, 1.0, 1.0, CFL[j])
        bc_grid = grid(0.0, 1.0, dx, 0.0, 1.0, dt)
        bc_grid = bc1(bc_grid)
#       print bc_grid
#       print tgrid
        title = "Scheme : "+ str(string1[i])+" CFL = " + str(CFL[j])
        print title
        bc_grid = string[i](bc_grid, xgrid, tgrid, CFL[j])
        print bc_grid

        plot_grid(bc_grid,0, xgrid, tgrid,title)


#Question2_1
string = [FTFS, FTCS, FTBS]
string1 = ["FTFS", "FTCS", "FTBS"]

xgrid = 100.0
CFL = [0.5 , 1 , 1.5]
for i in range(3):
    for j in range(3):
        tgrid,dt,dx = timestep(0.0, 1.0, xgrid, 0.0, 1.0, 1.0, CFL[j])
        bc_grid = grid(0.0, 1.0, dx, 0.0, 1.0, dt)
        print len(bc_grid[0,:]) , len(bc_grid[:,0])
        bc_grid = bc2_1(bc_grid, xgrid)
        #print bc_grid
        #print tgrid
        title = "Scheme : "+ str(string1[i])+" CFL = " + str(CFL[j])
        bc_grid = string[i](bc_grid, xgrid, tgrid, CFL[j])
        #print bc_grid

        plot_grid(bc_grid, 0,xgrid, tgrid,title)


#Question2_2
string = [FTFS, FTCS, FTBS]
string1 = ["FTFS","FTCS", "FTBS"]

xgrid = 100.0
CFL = [0.5 , 1 , 1.5]
for i in range(3):
    for j in range(3):
        tgrid,dt,dx = timestep(0.0, 1.0, xgrid, 0.0, 1.0, 1.0, CFL[j])
        bc_grid = grid(0.0, 1.0, dx, 0.0, 1.0, dt)
        bc_grid = bc2_2(bc_grid, xgrid)
#       print bc_grid
#       print tgrid
        title = "Scheme : "+ str(string1[i])+" CFL = " + str(CFL[j])
        bc_grid = string[i](bc_grid, xgrid, tgrid, CFL[j])
        print bc_grid

        plot_grid(bc_grid,0, xgrid, tgrid,title)

'''

#Question3
xgrid = 40
tgrid,dt,dx = timestep(-1.0, 1.0, xgrid, 0.0, 30.0, 1.0, 0.8)
print tgrid, xgrid , dt , dx
bc_grid = grid(-1.0, 1.0, dx, 0.0, 30.0, dt)
bc_grid[:,1:] = 0
bc_grid[:,0] = -np.sin(np.pi*bc_grid[:,0])
print bc_grid
#print tgrid
title = "Scheme : FTCS2 "+" CFL = "+ str(0.8)
bc_grid = FTCS2(bc_grid, xgrid, tgrid, 0.8)
#print bc_grid
#plot_grid(bc_grid,-1, xgrid, tgrid,title)
plt.plot(np.linspace(-1,1,39), bc_grid[:,30])
plt.title(title)
plt.show()



#Question3_ftbs
xgrid = 40
tgrid,dt,dx = timestep(-1.0, 1.0, xgrid, 0.0, 30.0, 1.0, 0.8)
print tgrid, xgrid , dt , dx
bc_grid = grid(-1.0, 1.0, dx, 0.0, 30.0, dt)
print len(bc_grid[0,:])
bc_grid[:,1:] = 0
bc_grid[:,0] = -np.sin(np.pi*bc_grid[:,0])
print len(bc_grid[0,:])
#print tgrid
title = "Scheme :FTBS2 "+" CFL = "+ str(0.8)
bc_grid = FTBS2(bc_grid, xgrid, tgrid, 0.8)
#print bc_grid
#plot_grid(bc_grid,-1, xgrid, tgrid,title)


plt.plot(np.linspace(-1,1,39), bc_grid[:,30])

plt.title(title)
plt.show()


#Quest3_2_ftcs

xgrid = 40
tgrid,dt,dx = timestep(-1.0, 1.0, xgrid, 0.0, 30.0, 1.0, 0.8)
print tgrid, xgrid , dt , dx
bc_grid = grid(-1.0, 1.0, dx, 0.0, 30.0, dt)
print len(bc_grid[0,:])
bc_grid[:,1:] = 0
xvalues = np.linspace(-1,1,40)
for i in range(0,len(xvalues)-1):
    if abs(xvalues[i]) < 1/3:
        bc_grid[i,0] = 1
    else:
        bc_grid[i,0] = 0
print len(bc_grid[0,:])
#print tgrid
title = "Scheme :FTCS2 "+" CFL = "+ str(0.8)
bc_grid = FTCS2(bc_grid, xgrid, tgrid, 0.8)
#print bc_grid
#plot_grid(bc_grid,-1, xgrid, tgrid,title)


plt.plot(np.linspace(-1,1,39), bc_grid[:,4])

plt.title(title)
plt.show()

#Quest3_2_ftcs

xgrid = 40
tgrid,dt,dx = timestep(-1.0, 1.0, xgrid, 0.0, 30.0, 1.0, 0.8)
print tgrid, xgrid , dt , dx
bc_grid = grid(-1.0, 1.0, dx, 0.0, 30.0, dt)
print len(bc_grid[0,:])
bc_grid[:,1:] = 0
xvalues = np.linspace(-1,1,40)
for i in range(0,len(xvalues)-1):
    if abs(xvalues[i]) < 1/3:
        bc_grid[i,0] = 1
    else:
        bc_grid[i,0] = 0
print len(bc_grid[0,:])
#print tgrid
title = "Scheme : FTBS2"+" CFL = "+ str(0.8)
bc_grid = FTBS2(bc_grid, xgrid, tgrid, 0.8)
#print bc_grid
#plot_grid(bc_grid,-1, xgrid, tgrid,title)

plt.plot(np.linspace(-1,1,39), bc_grid[:,4])

plt.title(title)
plt.show()



#Quest3_3_ftcs

xgrid = 600
tgrid,dt,dx = timestep(-1.0, 1.0, xgrid, 0.0, 30.0, 1.0, 0.8)
print tgrid, xgrid , dt , dx
bc_grid = grid(-1.0, 1.0, dx, 0.0, 30.0, dt)
print len(bc_grid[0,:])
bc_grid[:,1:] = 0
xvalues = np.linspace(-1,1,600)
for i in range(0,len(xvalues)-1):
    if abs(xvalues[i]) < 1/3:
        bc_grid[i,0] = 1
    else:
        bc_grid[i,0] = 0
print len(bc_grid[0,:])
#print tgrid
title = "Scheme : "+" CFL = "+ str(0.8)
bc_grid = FTCS2(bc_grid, xgrid, tgrid, 0.8)
#print bc_grid
#plot_grid(bc_grid,-1, xgrid, tgrid,title)

plt.plot(np.linspace(-1,1,599), bc_grid[:,4])

plt.title(title)
plt.show()

plt.plot(np.linspace(-1,1,599), bc_grid[:,40])

plt.title(title)
plt.show()
print bc_grid[:,4]
print bc_grid[:,40]

#Quest3_2_ftbs

xgrid = 600
tgrid,dt,dx = timestep(-1.0, 1.0, xgrid, 0.0, 30.0, 1.0, 0.8)
print tgrid, xgrid , dt , dx
bc_grid = grid(-1.0, 1.0, dx, 0.0, 30.0, dt)
print len(bc_grid[0,:])
bc_grid[:,1:] = 0
xvalues = np.linspace(-1,1,600)
for i in range(0,len(xvalues)-1):
    if abs(xvalues[i]) < 1/3:
        bc_grid[i,0] = 1
    else:
        bc_grid[i,0] = 0
print len(bc_grid[0,:])
#print tgrid
title = "Scheme : "+" CFL = "+ str(0.8)
bc_grid = FTBS2(bc_grid, xgrid, tgrid, 0.8)
#print bc_grid
#plot_grid(bc_grid,-1, xgrid, tgrid,title)


plt.plot(np.linspace(-1,1,599), bc_grid[:,4])

plt.title(title)
plt.show()

plt.plot(np.linspace(-1,1,599), bc_grid[:,40])

plt.title(title)
plt.show()
print bc_grid[:,4]
print bc_grid[:,40]



