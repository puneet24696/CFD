import numpy as np
import numba

#import hope
import matplotlib.pyplot as plt

e = 1
while e+1>1:
    e=e*0.5
print e

def grid(n):
    x,y = np.mgrid[0:1:n*1j,0:1:n*1j]
    return x,y

#print grid(n)


def bc(x,y):
    phi = x**2 - y**2
    phi[1:-1,1:-1]=0
    return phi



def jaco( n , phi ,prephi , w , N  ):
    phi[1:-1,1:-1] = 0.25*(phi[2:,1:-1]+phi[0:-2,1:-1]+phi[1:-1,2:]+phi[1:-1,0:-2])
    return phi

jacojit = numba.jit(jaco)


def gau(n,phi, prephi , w ,N ):

    for l in range(1,n-1):
        for m in range(1,n-1):
            phi[l,m] = 0.25*(phi[l+1,m] + phi[l-1,m] + phi[l,m+1] + phi[l,m-1])


    return phi


gaujit = numba.jit(gau)

def errfunc(n, err,  phi,phipre):
    err = ((((sum(sum((phi-phipre)**2))))**0.5)/n)
    return err


def sor(n , phi, phipre , w , N ):
    for l in range(1,n-1):
        for m in range(1,n-1):
            phi[l,m] = 0.25*(phi[l+1,m] + phi[l-1,m] + phi[l,m+1] + phi[l,m-1])
            phi[l,m] = (1-w)*phipre[l,m] + w*phi[l,m]

    return phi

sorjit = numba.jit(sor)



def plot1(w , err  , q , name,xlabl , ylabl , z  ):
    plt.plot(w,err )
    plt.text(w[-1 * z],err[-1 *z], name)
    plt.xlabel(xlabl)
    plt.ylabel(ylabl)
    str1 = 'sampleplot' + str(q) + '.png'
    plt.savefig(str1)
    plt.show()


def resifunc(n ,  phi):
    resi = (sum(sum((phi[1:-1,1:-1] - 0.25*(phi[2:,1:-1]+phi[0:-2,1:-1]+phi[1:-1,2:]+  phi[1:-1,0:-2]))**2)))**0.5/(n)

    return resi

def solver(n , func , w, N ):

    if func == jacojit or func ==  gaujit or func == gau or func == jaco:
    #for gauss and jacobian
        err1 = [10]

        count = 0
        x,y = grid(n)
        phi = bc(x,y)
        while err1[-1]> 2*e and count < 100000:

            phipre = phi.copy()
            phi = func(n, phi , phipre ,  w, N )
            err1.append(  errfunc(n, 0 , phi, phipre))
        err1.pop(0)

        resi1 = [10]
        count = 0
        x,y = grid(n)
        phi = bc(x,y)
        while resi1[-1] > 2*e and count < 100000:
            phipre = phi.copy()
            phi = func ( n , phi , phipre , w, N )
            resi1.append(resifunc( n ,    phi ))

        return err1 , resi1



    elif func == sor or func == sorjit:
    #for SOR
        err = []
        resi = []
        lasterr = []
        lastresi = []
        for j in w:
            x,y = grid(n)
            phi = bc(x,y)

            for i in range(N):
                phipre = phi.copy()
                phi = func ( n , phi, phipre , j , N )
                err.append(errfunc(n , err , phi , phipre ))
                resi.append(resifunc(n ,  phi))
            lasterr.append( err[-1])
            lastresi.append( resi[-1])

        return lasterr,lastresi


#print (solver(41,sorjit,0.1 , 20  ))



#quest 1
print 'quest1'
n = [11,21,41,101]
for i in n:
    err , resi = solver(i, jacojit ,1,1)
    err = np.log(np.array(err))
    itr = np.linspace(1,len(err),len(err))
    plot1(itr,err,1.1,'jacobi error' , 'iteration' , 'log of error/residue', 2)

    resi = np.log(np.array(resi))
    itr2 = np.linspace(1,len(resi),len(resi))
    plot1(itr2,resi,1.1,'jacobi residue' , 'iteration' , 'log of error/residue', 100)
plt.figure()
for i in n:

    err, resi = solver(i,gaujit,1,1)
    itr = np.linspace(1,len(err),len(err))
    err = np.log(np.array(err))
    resi =  np.log(np.array(resi))
    itr2 = np.linspace(1,len(resi),len(resi))
    plot1(itr,err,1.2,'gauss error' , 'iteration' , 'log of error/residue',2)
    plot1(itr2,resi ,1.2,'gauss residue' , 'iteration' , 'log of error/residue',100)


#quest 2 and 3
N = [20,50,100]
minw= []
print 'quest 2 and 3 '
for i in N:
    w = np.linspace(0.1,1.9,19)
    lasterr,lastresi = solver(41, sorjit , w , i)
    minwe =  lasterr.index(min(lasterr))
    minwr = lastresi.index(min(lastresi))
    print w[minwe]
    print w[minwr]
    plt.figure()
    plot1(w,np.log(lasterr), 2+0.1*N.index(i),'SOR err for n =%s' % str(i), 'omega' , 'log of error/residue', 2)
    plot1(w,np.log(lastresi), 2+0.1*N.index(i),'SOR resi for n =%s' % str(i), 'omega' , 'log of error', 4)

#answer w  = 0.1 ,1.8,1.8



#quest 4
w= np.linspace(1.7,1.9,21)
lasterr, lastresi = solver(41 , sorjit , w , 50 )
plt.figure()
plot1(w,np.log(lasterr),4,'SORerr' , 'omega' , 'log of error/residue',2)
plot1(w,np.log(lastresi),4,'SORresi' , 'omega' , 'log of error/residue',4)
minwe = lasterr.index(min(lasterr))
print 'quest4'
print w[minwe]
minwr = lastresi.index(min(lastresi))
print w[minwr]
#answer =1.79



#quest 5
w = np.linspace(0.1,1.9,19)
lasterr,lastresi = solver ( 101 , sorjit , w , 100)
plt.figure()
plot1(w,np.log(lasterr),5,'SORerr' , 'omega' , 'log of error',2)
minwe = lasterr.index(min(lasterr))
plot1(w,np.log(lastresi),5,'SORresi' , 'omega' , 'log of error',4)
minwr = lastresi.index(min(lastresi))
print 'quest5'
print w[minwe]
print w[minwr]
#answer = 1.9


#quest6
#for SOR
print 'quest 6'
n =101
w = np.linspace(1,2,11)
lasterr = []
lastresi = []
for j in w:
    count = 0
    err = [10]
    x,y = grid(n)
    phi = bc(x,y)
    while err[-1] > 3* e and count < 30000:
        phipre = phi.copy()
        phi = sorjit( n , phi, phipre , j , 1 )
        err.append(errfunc(n , 0 , phi , phipre ))
        count = count +1
    lasterr.append(count)

for j in w:
    count = 0
    resi = [10]
    x,y = grid(n)
    phi = bc(x,y)
    while resi[-1] > 3* e and count < 30000:
        phipre = phi.copy()
        phi = sorjit( n , phi, phipre , j , 1 )
        resi.append(resifunc(n ,  phi  ))
        count = count +1
    lastresi.append(count)
plt.figure()
plot1(w ,lasterr , 6,'SORerr' ,'w' , 'iteration',2)
plot1(w,lastresi , 6, 'SORresi' , 'w' , 'iteration' , 4)

print(lasterr)
print(lastresi)
minwe = lasterr.index(min(lasterr))
print w[minwe]
minwr = lastresi.index(min(lastresi))
print w[minwr]
#answer = 1.8



#quest 7
print 'quest 7'
n=101
plt.figure()
err,resi = solver(n ,jacojit , 1,1)
itr = np.linspace(1,len(err),len(err))
plot1(itr,np.log(np.array(err)),7.1,'jacoberr','iteration' , 'log of error/residue',2)
itr2 = np.linspace(1,len(resi),len(resi))
plot1(itr2,np.log(np.array(resi)),7.1,'jacobresi','iteration' , 'log of error/residue',4)




err,resi = solver( n , gaujit , 1,1)
itr = np.linspace(1,len(err),len(err))
plot1(itr,np.log(np.array(err)),7.1, 'gauserr','iteration' , 'log of error/residue',2)
itr2 = np.linspace(1,len(resi),len(resi))
plot1(itr2,np.log(np.array(resi)),7.1,'guesresi','iteration' , 'log of error/residue', 4)



count=0
err = [10]

x,y = grid(n)
phi=bc(x,y)

while err[-1] > 3*e and count < 30000:
    phipre = phi.copy()
    phi = sorjit( n , phi, phipre , w[minwe] , 1 )
    err.append(errfunc(n , err , phi , phipre ))
    count = count +1
plot1(np.linspace(1,len(err),len(err)),np.log(np.array(err)),7.1 , 'sorerr','iteration' , 'log of error/residue', 2)

count=0
resi = [10]

x,y = grid(n)
phi=bc(x,y)

while resi[-1] > 3*e and count < 30000:
    phipre = phi.copy()
    phi = sorjit( n , phi, phipre , w[minwr] , 1 )
    resi.append(resifunc(n ,  phi  ))
    count = count +1
plot1(np.linspace(1,len(resi),len(resi)),np.log(np.array(resi)),7.1 , 'sorresi','iteration' , 'log of error/residue', 10)




