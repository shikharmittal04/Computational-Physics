#Shikhar Mittal (DTP)
#Jocobi, Gauss-Seidel, relaxation and Conjugate gradient method for solving AX=b equation.

import numpy as np
import array as arr

x_true=np.array([7.859713071,0.422926408,-0.073592239,-0.540643016,0.010626163])

A=np.array([[0.2, 0.1,1, 1,0],[0.1,4, -1, 1, -1],[1, -1, 60, 0, -2],[ 1,1,0, 8,4],[0, -1, -2, 4, 700]])
b=np.array([1,2,3,4,5])
###############################################################
#Jacobi 

c_J=0							#To store the no.of iteration required
D=np.ones(5)					#Difference of true and approx. sol.
x_old=np.zeros(5)				#Current iteration approx. sol.
x_new=np.zeros(5)				#Next iteration approx. sol.
while np.size(D[abs(D)>0.01])!=0:
    for i in range(5):
        Sum=0
        q=arr.array('i',range(5))
        q.remove(i)
        for j in q:
            Sum=Sum+A[i,j]*x_old[j]

        x_new[i]=(b[i]-Sum)/A[i,i]
    x_old=x_new
    D=x_true-x_old
    c_J+=1

print('By Jacobi method:\nx =',x_new)
print('No.of iterations for Jacobi method = ',c_J)
################################################################
#Gauss-Seidel

c_GS=0
D=np.ones(5)
x=np.zeros(5)
while np.size(D[abs(D)>0.01])!=0:
    for i in range(5):
        Sum=0
        q=arr.array('i',range(5))
        q.remove(i)
        for j in q:
            Sum=Sum+A[i,j]*x[j]

        x[i]=(b[i]-Sum)/A[i,i]
        
    D=x_true-x
    c_GS+=1

print('\nBy Gauss-Seidel method:\nx =',x)
print('No.of iterations for Gauss-Seidel method = ',c_GS)
#################################################################
#Relaxation
w=1.25
c_R=0
D=np.ones(5)
x=np.zeros(5)
while np.size(D[abs(D)>0.01])!=0:
    for i in range(5):
    	res=b[i]-sum(A[i]*x)
    	x[i]=x[i]+w*res/A[i,i]
	
    D=x_true-x
    c_R+=1

print('\nBy relaxation method:\nx =',x)
print('No.of iterations for relaxation method = ',c_R)

################################################################
#Conjugate gradient method

c_CG=0
D=np.ones(5)
x=np.zeros(5)
r = b - np.dot(A, x)
p = r
Norm_old = np.dot(np.transpose(r), r)

while np.size(D[abs(D)>0.01])!=0:
    alp = Norm_old / np.dot(np.transpose(p), np.dot(A, p))
    r = r - alp*np.dot(A, p)
    x = x + alp*p
    Norm_new = np.dot(np.transpose(r), r)
    p = r + (Norm_new/Norm_old)*p
    Norm_old = Norm_new
    D=x_true-x
    c_CG+=1
    
print('\nBy Conjugate gradient method:\nx =',x)
print('No.of iterations for Conjugate gradient method = ',c_CG)
