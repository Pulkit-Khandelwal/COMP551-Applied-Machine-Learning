# =========================== IMPORTS ===============================
import numpy as np
from numpy.linalg import inv

# =========================== TESTING ===============================
X = np.array([[0.86], [0.09], [-0.85], [0.87], [-0.44], [-0.43],
              [-1.10], [0.40], [-0.96], [0.17]])

Y = np.array([[2.49], [0.83], [-0.25], [3.10], [0.87], [0.02],
              [-0.12], [1.81], [-0.83], [0.43]])

# closed-form solution
Xarg = np.insert(X, 1, 1, axis=1)
temp1 = np.dot(Xarg.T, Xarg)
temp2 = np.dot(Xarg.T, Y)

w = inv(temp1).dot(temp2)

# gradient-descent method
w_gd = np.array([[0], [0]])
alpha = 0.1

delta_err = np.dot(np.dot(Xarg.T, Xarg), w_gd) - np.dot(Xarg.T, Y)
# print delta_err

for k in range(0, 100):
    w_gd = w_gd - alpha * delta_err
    delta_err = np.dot(np.dot(Xarg.T, Xarg), w_gd) - np.dot(Xarg.T, Y)
    cost = sum(sum(Y - np.dot(Xarg, w_gd)))

print w_gd
print ""
print np.dot(Xarg, w_gd)
