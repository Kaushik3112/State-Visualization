import numpy as np
from numba import njit
from scipy.linalg import expm


OmegaX = np.array([[0, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]], dtype=np.complex64)

OmegaY = np.array([[0, 0, 1],
                  [0, 0, 0],
                  [-1, 0, 0]], dtype=np.complex64)

OmegaZ = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=np.complex64)


disc = 1280
n = 6
v_n = 1500/2
w_n = v_n/4
t = np.pi/(2*w_n)
t_list = np.linspace(0, t, disc)


@njit
def cossingen(t, v_list, j, n):
    temp_var = 1

    if j > 0:
        for i in range(j):
            temp_var = 2*temp_var*np.cos(v_list[0, i]*t)
    if j == (n-1):
        temp_var = 2*temp_var*np.cos(v_list[0, j]*t)
    else:
        temp_var = 2*temp_var*np.sin(v_list[0, j]*t)

    return temp_var


@njit
def v_gen(t, w_list, v_list, n):
    temp_var = 0

    for i in range(n):
        temp_var = temp_var + w_list[0, i+1]*cossingen(t, v_list, i, n)

    return temp_var


@njit
def Ham_gen(t, n, w_0, w_list, v_list, OmegaX, OmegaY, OmegaZ):
    Ham = (w_0*OmegaZ + w_list[0, 0]*OmegaX +
           v_gen(t, w_list, v_list, n)*OmegaY)
    return Ham


@njit
def t_mod(pulse, t, t_range):
    if pulse == "pi":
        t = t*2
        t_range = t_range*2
    elif pulse == "pi/4":
        t = t/2
        t_range = t_range/2
    else:
        pass

    return t, t_range


def propagation(v_list, w_0, w_list, M0, technique,
                pulse, t_range, disc, n, t):

    t, t_range = t_mod(pulse, t, t_range)

    if technique == "Unitary":
        U = np.eye(3, dtype=np.complex64)
        for i in range(disc):
            temp = Ham_gen(t_range[i], n, w_0, w_list,
                           v_list, OmegaX, OmegaY, OmegaZ)
            U = expm(temp*(t_range[1]-t_range[0]))@U

        A = U@M0

    return A


def propagation_w_state(v_list, w_0, w_list, M0,
                        technique, pulse, t_range, disc, n, t):

    t, t_range = t_mod(pulse, t, t_range)
    state_list = M0
    state = M0

    if technique == "Unitary":
        U = np.eye(3, dtype=np.complex64)
        for i in range(disc):
            temp = Ham_gen(t_range[i], n, w_0, w_list,
                           v_list, OmegaX, OmegaY, OmegaZ)
            U = expm(temp*(t_range[1]-t_range[0]))@U
            state = U@M0
            state_list = np.append(state_list, state, axis=1)

        A = U@M0

    return A, state_list


@njit
def multi_pulse_prop(v_list, c_list, w_list, M0, technqiue, pulse_list):

    A = np.eye(3, dtype=np.complex64)
    for pulse in pulse_list:
        M = propagation(v_list, c_list, w_list, M0, technqiue, pulse)
        A = M@A

    return A
