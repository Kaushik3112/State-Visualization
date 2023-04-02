import numpy as np
from numba import njit


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
    return (w_0*OmegaZ + w_list[0]*OmegaY + v_gen(t, w_list, v_list, n)*OmegaY)


@njit
def propagation(v_list, c_list, w_list, M0, technique, pulse):

    if technique == "Unitary":
        A = 0     # place holder

    elif technique == "ODE44":
        A = 1

    return A

# @njit
# def multi_pulse_prop(v_list, c_list, w_list, M0, technqiue, pulse_list):

#     I = np.eye(3, dtype = np.complex64)
#     A = I
#     for pulse in pulse_list:
#         M = propagation(v_list, c_list, w_list, M0, technqiue, pulse)
#         A = M@A


#     return A
