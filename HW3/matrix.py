import numpy as np

m1 = np.array(
[[1, 1, 1, 1, 1],
 [4, 1, 6, 18, 8]]
)

#result = np.linalg.inv(m1.T @ m1) @ (m1.T) @ np.array([3.5, 1.0, 3.8, 10.1, 8.5]).T

I = np.identity(2)
m2 = np.linalg.inv(m1 @ m1.T) 
m3 = m2 @ m1 @ np.array([3.5, 1.0, 3.8, 10.1, 8.5]).T


m2_ridge = np.linalg.inv((m1 @ m1.T) + I)
print(m2_ridge @ m1 @ np.array([3.5, 1.0, 3.8, 10.1, 8.5]).T)