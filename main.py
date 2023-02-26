import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as am
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.cm as cm
import pickle
import os
from datetime import datetime
from scipy.fft import ifft2, fft2
import agent
import analyse
import visualise




def todo(tmax, sigma, eta, bc, grad, dim):
    t_max = tmax
    length = 128
    # length= 64
    if dim == '2D':
        height = 128
        model = agent.Kuramoto(t_max, height, length, sigma, eta, bc, grad)
        model.solve(dim=2)
        print('2D solved')
        model.save()
        visualise.plot2D_frame(model.tmax - 2, model)
        print('2D saved')
    elif dim == 'q2D':
        height = 20
        model = agent.Kuramoto(t_max, height, length, sigma, eta, bc, grad)
        model.solve(dim=2)
        print('2D solved')
        model.save()
        visualise.plot2D_frame(model.tmax - 2, model)
        print('2D saved')
    elif dim == '1D':
        model = agent.Kuramoto(t_max, length, 1, sigma, eta, bc, grad)
        model.solve()
        print('1D solved')
        model.save()
        visualise.plot1D_frame(model,t = -2)
        print('1D saved')
    else:
        print('Error shape not present')
        return
    return model.title,model


def custom():
    custom = agent.Kuramoto(1000, 20, 128, 0.1, 0.5, 'custom', [0, 0])
    custom.omegas[10][64] = 2
    custom.solve(2)
    custom.save()
    visualise.animate(custom, False)


def remove_harmonics(filename):
    analyse.remove_harmonic(agent.read(filename), agent.set_model(filename))


tmax = 1000
vals = [0,0.05,0.1,0.2,1]

bcs = ['fix','grad','grad','grad','grad','periodic']
grad = [[0, 0], [1, 1], [-1, 1], [1, -1], [0, 0], []]
dims = ['1D','q2D','2D']


for i in [0]:
    for j in [0.2]:
        for k in [0]:
            for l in [1]:
                #title = todo(tmax, i, j, bcs[k], grad[k], dims[l])
                #anim_read(title, 'phase')
                #anim_read(title, 'vorticity')
                # remove_harmonic('20230208_c0_0.5_1')
                pass


#egbranch = agent.Kuramoto(200,20,128,0.1,0.5,'grad',[1,1],'branch')
#egbranch.solve(dim=3)
#print('solved')
#visualise.plot_branch(190,egbranch)
#visualise.animate(egbranch,False,'branch')
#plot1D_frame(eg1D,-2,False)
#plt.show()



model = todo(100,0.1,0.02,'fix',[0,0],'q2D')[1]
visualise.animate(model,False,'phase')
#model.find_vorticity()
#visualise.animate(model,False,'vorticity')
#analyse.plot_vd(model)
