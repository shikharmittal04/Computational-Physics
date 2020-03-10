#Shikhar Mittal (DTP)
#Eigenproblem part-I

import numpy as np
A=np.array([[5, -2], [-2, 8]])				#Given symmetric matrix A
B=A											#'A' will get changed, hence save 'A' in a new variable.
M=np.identity(2,dtype=float)				#This will store the product of Q's
C=int(input('No.of iterations: '))			#No.of times we perform QR decomposition
count=0
while count<C:
	Q,R=np.linalg.qr(A)
	A=np.dot(R,Q)
	M=np.dot(M,Q)
	count=count+1

MT=M.transpose()
D=np.dot(np.dot(MT,B),M)

np.set_printoptions(precision=2)   			#Rounds up to 2 decimals
np.set_printoptions(suppress=True)  		#NOT in scientific notation

print("Eigenvalues using QR decomposition = ", end =" ")
for i in range(len(B)):
	print("{:.2f}".format(D[i][i]), end =" ")

lam,V=np.linalg.eigh(B)						#eigh to be used only for symmetric matrix.
print("\nEigenvalues using 'linalg.eigh':",lam)
