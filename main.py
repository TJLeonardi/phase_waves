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


def todo(tmax, sigma, eta, bc, grad, dim,init):
    t_max = tmax
    length = 128
    # length= 64
    if dim == '2D':
        height = 128
        model = agent.Kuramoto(t_max, height, length, sigma, eta, bc, grad,dim,init)
        model.solve(dim=2)
        print('2D solved')
        model.save()
        visualise.plot2D_frame(model.tmax - 2, model)
        print('2D saved')
    elif dim == 'q2D':
        height = 20
        model = agent.Kuramoto(t_max, height, length, sigma, eta, bc, grad,dim,init)
        model.solve(dim=2)
        print('2D solved')
        model.save()
        visualise.plot2D_frame(model.tmax - 2, model)
        print('2D saved')
    elif dim == '1D':
        model = agent.Kuramoto(t_max, length, 1, sigma, eta, bc, grad,dim,init)
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


tmax = 100
vals = [0,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5]

bcs = ['fix','grad','grad','grad','grad','periodic']
grad = [[0, 0], [1, 1], [-1, 1], [1, -1], [0, 0], []]
dims = ['1D','q2D','2D']


for i in vals:
    for j in vals:
        for k in [4]:
            for l in [1]:
                meas = [0,0,0]
                for m in range(5):
                    model = todo(tmax, vals[i], vals[j], bcs[k], grad[k], 'q2D','rand')[1]
                    guys =  analyse.plot_vd(model)
                    print(guys)
                    meas[0] += guys[0]
                    meas[1]+= guys[1]

                    meas[2] += analyse.interdistances_multiframe(model, 0, model.tmax)
                    print(meas)
                    if m %5 == 4:
                        visualise.animate(model,False,'phase')
                        analyse.pd_file(model,np.divide(meas,5))



#egbranch = agent.Kuramoto(200,20,128,0.1,0.5,'grad',[1,1],'branch')
#egbranch.solve(dim=3)
#print('solved')
#visualise.plot_branch(190,egbranch)
#visualise.animate(egbranch,False,'branch')
#plot1D_frame(eg1D,-2,False)
#plt.show()


#model = todo(500,0.3,0.5,'grad',[0,0],'q2D','rand')[1]
#model = agent.set_model('data_q2D_D_0.6_1.1_0')
#visualise.animate(model,False,'phase')
#analyse.plotvcorr(model)
#analyse.plotpcorr(model)
#reps = agent.Repeats(5,10000,128,1,0,0,'grad',[1,1],'1D')
#reps.create()
#reps.averagetheta()
#visualise.plot_avg(reps,99)
#visualise.animate(model,False,'phase')
#visualise.animate(model,False,'vorticity')
#analyse.plot_vd(model)
#analyse.vorsites(model,179)
#analyse.interdistances_multiframe(model,0,model.tmax)
#analyse.plot_m(model)
