import numpy as np
import time

class Least_Square:
    """
        X_train: an 2D matrix (m,n)
        y_train: an 1D matrix (m,1)
        f(x) == ||Ax-b||_2
        f(x) = (x_T A_T -b)(Ax -b)
        f(x) = x_T A_T Ax - 2 b_T Ax + b_T b
        B = A_T A
        C_T = 2 b_T A
        d = b_T b
        f(x) = x_T Bx - C_T x + d
        f_prime(x) = 2Bx - C
    """
    def __init__(self, X_train, y_train):
        self.A = X_train
        self.b = y_train
        self.B = self.A.T @ self.A
        self.C_T = 2 * self.b.T @ self.A
        self.C = self.C_T.T
        self.d = self.b.T @ self.b
        self.m, self.n = self.A.shape
        L_F = np.linalg.norm(self.A)**2
        self.hess = (2/self.m)*self.B
    def F(self, x):
            return (x.T @ self.B @ x - self.C_T @ x + self.d)/self.m

    def grad_F(self, x):
            return (2 * self.B @ x - self.C)/self.m