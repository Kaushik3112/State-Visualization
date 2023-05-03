import matplotlib as mpl
import numpy as np
from pylab import *
from qutip import *
from matplotlib import cm
import math as m
import imageio

def state_converter(state_list):
    # thetas = np.arctan((np.sqrt(state_list[0]*state_list[0] + state_list[1]*state_list[1]))/state_list[2])
    # phis = np.arctan(state_list[1]/state_list[0])
    thetas = np.zeros_like(state_list[0], dtype=np.complex64)
    phis = np.zeros_like(thetas, dtype=np.complex64)
    for i in range(len(thetas)):
        thetas[i] = m.atan2(state_list[2, i], m.sqrt(state_list[0, i]*state_list[0, i] + state_list[1, i]*state_list[1, i]))
        phis[i] = m.atan2(state_list[1, i], state_list[0, i])
    phis[np.isnan(phis)] = 0

    states = []

    for i in range(len(thetas)):
        states.append((np.cos(thetas[i]/2)*basis(2,0) +
                      (np.cos(phis[i]) + 1j*np.sin(phis[i]))*np.sin(thetas[i]/2)*basis(2,1)).unit())
    return states

def animate_bloch(states, duration=0.1, save_all=True, path="tmp"):
    
    b = Bloch()

    b.vector_color = ['r']

    b.view = [-40,30]

    images=[]

    try:

        length = len(states)

    except:

        length = 1

        states = [states]

    ## normalize colors to the length of data ##

    nrm = mpl.colors.Normalize(0,length)

    colors = cm.cool(nrm(range(length))) # options: cool, summer, winter, autumn etc.

    ## customize sphere properties ##

    b.point_color = list(colors) # options: 'r', 'g', 'b' etc.

    b.point_marker = ['o']

    b.point_size = [30]

    for i in range(0, length, 10):

        b.clear()

        b.add_states(states[i])

        b.add_states(states[:(i+1)],'point')

        if save_all:

            b.save(dirc=path) #saving images to tmp directory

            filename="tmp/bloch_%01d.png" % i

        else:

            filename='temp_file.png'

            b.save(filename)
