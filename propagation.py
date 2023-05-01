import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from initialization import v_list_gen, c_list_gen, w_list_gen


OmegaX =np.array([[0, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]], dtype=np.complex64)

OmegaY =np.array([[0, 0, 1],
                  [0, 0, 0],
                  [-1, 0, 0]], dtype=np.complex64)

OmegaZ =np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=np.complex64)


disc = 128000
n = 6
v_n = 1500/2
w_n = v_n/4
t = np.pi/(2*w_n)
t_list = np.linspace(0, t, disc)

@njit
def cossingen(t, w_list, j, n):
    temp_var = 1

    if j > 1:
        for i in range(j-1):
            temp_var = 2*temp_var*np.cos(w_list[i]*t)
    elif j == n:
        temp_var = 2*temp_var*np.cos(w_list[j]*t)
    else:
        temp_var = 2*temp_var*np.sin(w_list[j]*t)

    return temp_var


@njit
def v_gen(t, w_list, v_list, n):
    temp_var = 0

    for i in range(n):
        temp_var = temp_var + w_list[i+1]*cossingen(t, w_list, i, n)

    return temp_var


@njit
def Ham_gen(t, n, w_0, w_list, v_list, OmegaX, OmegaY, OmegaZ):
    return (w_0*OmegaZ + w_list[0]*OmegaX + v_gen(t, w_list, v_list, n)*OmegaY)

@njit
def ode_function(t, v, n, w_0, w_list, v_list, OmegaX, OmegaY, OmegaZ):
    Ham = Ham_gen(t, n, w_0, w_list, v_list, OmegaX, OmegaY, OmegaZ)
    vout = Ham@v
    return vout

@njit
def propagation(v_list, w_list, M0, technique, pulse, t_range, disc, n):

    if technique == "Unitary":
        U = np.eye(3, dtype=np.complex64)
        for i in range(disc):
            temp = Ham_gen(t_range[i], n, w_list[0], w_list, v_list, OmegaX, OmegaY, OmegaZ)
            U = np.expm(temp*(t_range[1]-t_range[0]))@U

        A = U@M0

    return A

@njit
def multi_pulse_prop(v_list, c_list, w_list, M0, technqiue, pulse_list):

    I = np.eye(3, dtype = np.complex64)
    A = I
    for pulse in pulse_list:
        M = propagation(v_list, c_list, w_list, M0, technqiue, pulse)
        A = M@A

    return A
