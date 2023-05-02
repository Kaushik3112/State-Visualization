import numpy as np
from initialization import v_list_gen, c_list_gen, w_list_gen
from propagation import propagation, propagation_w_state
from argparse import ArgumentParser
import sys


def initial(n):
    k = n - 6
    v_n = 1500/(2**(k//2+1))
    w_n = v_n/4
    c_n = v_n/100
    w_max = (2**(n-6))*50000
    return v_n, w_n, c_n, w_max


if __name__ == "__main__":

    parser = ArgumentParser(prog="state visualization",
                            description="state visualization")
    parser.add_argument("-n", "--number", type=int, default=6)
    parser.add_argument("-s", "--state-req", type=bool, default=False)
    parser.add_argument("-o", "--output", type=str)

    args = parser.parse_args()

    OmegaX = np.array([[0, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]], dtype=np.complex64)

    OmegaY = np.array([[0, 0, 1],
                       [0, 0, 0],
                       [-1, 0, 0]], dtype=np.complex64)

    OmegaZ = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype=np.complex64)
    M0 = np.array([[0],
                   [0],
                   [1]], dtype=np.complex64)

    n = args.number

    v_n, w_n, c_n, w_max = initial(n)
    n_pieces = 100
    nu_list = v_list_gen(v_n, n)
    c_list = c_list_gen(c_n, n, nu_list)
    w_list = w_list_gen(w_n, n, nu_list, c_list)
    w_0 = np.linspace(-w_max, w_max, n_pieces)
    t_pulse = np.pi/(2*w_n)
    disc = 1280
    t_range = np.linspace(0, t_pulse, disc)
    w_list = w_list

    X3W = np.zeros(n_pieces)
    X2W = np.zeros(n_pieces)
    X1W = np.zeros(n_pieces)

    if args.state_req is True:
        for i in range(n_pieces):
            print(f"[{i}/{n_pieces}]", end="\r")
            if i == 50:
                X, state_list = propagation_w_state(v_list=nu_list, w_0=w_0[i],
                                                w_list=w_list, M0=M0,
                                                technique="Unitary",
                                                pulse="pi/2", t_range=t_range,
                                                disc=disc, n=n, t=t_pulse)
            else:
                X = propagation(v_list=nu_list, w_0=w_0[i], w_list=w_list, M0=M0,
                            technique="Unitary", pulse="pi/2", t_range=t_range,
                            disc=disc, n=n, t=t_pulse)
            
            X1W[i] = np.abs(X[0, 0])
            X2W[i] = np.abs(X[1, 0])
            X3W[i] = np.abs(X[2, 0])
    
    else:
        for i in range(n_pieces):
            print(f"[{i}/{n_pieces}]", end="\r")
            X = propagation(v_list=nu_list, w_0=w_0[i], w_list=w_list, M0=M0,
                            technique="Unitary", pulse="pi/2", t_range=t_range,
                            disc=disc, n=n, t=t_pulse)
            X1W[i] = np.abs(X[0, 0])
            X2W[i] = np.abs(X[1, 0])
            X3W[i] = np.abs(X[2, 0])

    print(state_list.shape)

    if args.output is not None:
        if args.state_req is True:
            np.savez(args.output, w_range=w_0,
                     X_final=X1W, Y_final=X2W,
                     Z_final=X3W, states=state_list)
        else:
            np.savez(args.output, w_range=w_0,
                     X_final=X1W, Y_final=X2W,
                     Z_final=X3W)
