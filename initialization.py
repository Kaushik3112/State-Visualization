import numpy as np
from numba import njit


@njit
def v_list_gen(v_n, n):
    v_list = np.zeros((1, n))
    v_list[0, n-1] = v_n
    for i in range(n-1):
        v_list[0, n-i-2] = v_list[0, n-i-1]*2

    return v_list


@njit
def c_list_gen(c_n, n, v_list):
    c_list = np.zeros((1, n+1))
    c_list[n] = c_n
    for i in range(n):
        c_list[0, n-1-i] = 2*np.sqrt(c_list[0, n-i]*v_list[0, n-1-i])
    
    return c_list


@njit
def w_list_gen(w_n, n, v_list, c_list):
    w_list = np.zeros((1, n+1))
    w_list[n] = w_n
    for i in range(n):
        w_list[i] = v_list[i] - c_list[i+1]

    return w_list
