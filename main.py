import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from initialization import v_list_gen, c_list_gen, w_list_gen
from propagation import cossingen, v_gen, Ham_gen, propagation, multi_pulse_prop
import matplotlib.pyplot as plt


OmegaX =np.array([[0, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]], dtype=np.complex64)

OmegaY =np.array([[0, 0, 1],
                  [0, 0, 0],
                  [-1, 0, 0]], dtype=np.complex64)

OmegaZ =np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=np.complex64)
M0=np.array([[0],
             [0],
             [1]], dtype=np.complex64)
n = 6
v_n = 1500/2
w_n = v_n/4
c_n = v_n/100
n_pieces = 100
nu_list = v_list_gen(v_n, n)
c_list = c_list_gen(c_n, n, nu_list)
w_list = w_list_gen(w_n, n, nu_list, c_list)
w_0 = np.linspace(-50000, 50000, n_pieces)
t_pulse = np.pi/(2*w_n)
disc= 128000
t_range = np.linspace(0, t_pulse, disc)
w_list = w_list

X3W= np.zeros((1,n_pieces))
X2W= np.zeros((1,n_pieces))
X1W= np.zeros((1,n_pieces))

for i in range(n_pieces):
    X = propagation(nu_list, w_0, w_list, M0, "Unitary", "pi/2", t_range, disc, n)
    X1W[i] = X[0,0]
    X2W[i] = X[0,1]
    X3W[i] = X[0,2]

plt.plot(X1W, w_0)