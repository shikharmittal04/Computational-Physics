#Shikhar Mittal (DTP)
#System of linear equations

import numpy as np
A=np.matrix('1 0.67 0.33;0.45 1 0.55;0.67 0.33 1')
b=np.matrix('2;2;2')
x=np.linalg.solve(A,b)
print('The solution is:\n',x)
