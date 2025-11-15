import numpy as np
from math import sqrt


M = 92
N = 32000
mu = 0
sigma = 1

# 随机矩阵 相当于N个近似正交的基
W = np.random.randn(M, N)/np.sqrt(M)
print(W.shape)
print(np.linalg.norm(W))

# 测试向量
V = np.random.normal(mu, sigma, (N,))

V_prime = np.dot(W, V)
# print(V_prime)

V_reconstruct = np.dot(np.linalg.pinv(W), V_prime)

print(np.linalg.norm(V))
print(np.linalg.norm(V_reconstruct))
print(np.linalg.norm(V-V_reconstruct))