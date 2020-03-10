#Shikhar Mittal
#Aim: To find the dominant eigenvalue using the power method
import numpy as np
A=np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])	#Given matrix
x=np.ones(3)								#Initial guess

lam_new=np.dot(np.transpose(np.dot(A,x)),x)/np.dot(np.transpose(x),x)
lam_old=0
c=0
while abs(lam_new-lam_old)>0.01:
	lam_old=lam_new
	x=np.dot(A,x)
	lam_new=np.dot(np.transpose(np.dot(A,x)),x)/np.dot(np.transpose(x),x)
	c+=1
	
print("Dominant eigenvalue: {:0.3f}".format(lam_new))
print("No.of iterations required to reach 1% accuracy:", c)
