# Shikhar Mittal
# Singular Value Decomposition
import numpy as np
from timeit import default_timer as timer
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)  #NOT in scientific notation

m=int(input('Enter no.of rows:'))
n=int(input('Enter no.of columns:'))
print("Enter the entries for A in a single line (separated by space): ") 
  
entries1 = list(map(int, input().split())) 
A = np.array(entries1).reshape(m, n)
print("The matrix entered is, A =\n",A)


start = timer()

W1=np.dot(np.transpose(A),A)
W2=np.dot(A,np.transpose(A))

E_V,V=np.linalg.eigh(W1)
idx = E_V.argsort()[::-1]   
E_V = E_V[idx]
V = V[:,idx]

E_U,U=np.linalg.eigh(W2)
idx = E_U.argsort()[::-1]   
E_U = E_U[idx]
U = U[:,idx]

S1=np.zeros(A.shape)
if (m>=n):
	S1[:n,:n]=np.diag(np.sqrt(E_V))
else:
	S1[:m,:m]=np.diag(np.sqrt(E_U))

print("\nU=",U)
print("S=",S1)
print("V=",V)
end = timer()
print("\nTime elapsed (in sec) without using linalg.svd = ",end - start)

##################################################################
start = timer()
u, s, vT = np.linalg.svd(A)
l=min(m,n)
S2=np.zeros(A.shape)                
S2[:l,:l]=np.diag(s)   				
print('\nUsing linalg.svd() we get:')
print('\nu=',u)
print('S=',S2)
print('v^T=',vT)
end = timer()
print("\nTime elapsed (in sec) using linalg.svd = ",end - start)

