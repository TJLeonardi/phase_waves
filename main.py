import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as am
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.cm as cm
import pickle
import os
from datetime import datetime
from scipy import signal
import agent
import analyse
import visualise


def todo(tmax, height,length,sigma, eta, bc, grad, dim,init):
    t_max = tmax
    #length = 64
    # length= 64
    if dim == '2D':
        #height = 64
        model = agent.Kuramoto(t_max, height, length, sigma, eta, bc, grad,dim,init)
        model.solve(dim=2)
        print('2D solved')
        model.save()
        visualise.plot2D_frame(model.tmax - 2, model)
        print('2D saved')
    elif dim == 'q2D':
        #height = 20
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


tmax = 500
vals = [0.1,0.2,0.3,0.4,0.5,0.6]
vals2 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]

#vals = [1.1,1,0.9]

bcs = ['fix','grad','grad','grad','grad','periodic']
grad = [[0, 0], [1, 1], [-1, 1], [1, -1], [0, 0], []]
dims = ['1D','q2D','2D']

def listoff():
    for i in range(len(vals)):
        for j in range(len(vals2)):
            for k in [5]:
                for l in [1]:
                    meas = [0,0,0]
                    for m in range(1):
                        #model = todo(tmax, 64,64,vals[i], vals2[j], bcs[k], grad[k], '2D','rand')[1]
                        #visualise.animate(model,False,'phase')
                        name = f'data_q2D_p_{vals[i]}_{vals2[j]}_0'
                        get_data(name)
                        #analyse.pd_file(model,np.divide(meas,5))

def phase_diagram():
    file = np.array(pickle.load(open('vortices/pd_q2D_periodic_5000.p', 'rb')))
    file2 = np.array(pickle.load(open('vortices/pd_2D_periodic_100.p', 'rb')))
    print(file.shape)
    print(file2.shape)
    #pd = np.concatenate([file,file2])


    titles = ['sigma','eta','vort_net_last', 'vort_ms_last', 'peak_number', 'r_last', 'm_last', 'v_last_mean', 'v_last_stdv']
    var = 5
    z = file.T[var]
    print(z)
    print(z.shape)

    #print(pd)
    #print(z[:160].reshape(8,15))
    plt.imshow(np.flip(z.reshape(6,15),0),interpolation=None,extent=[0.1,1.5,0.1,0.6])
    plt.xlabel('eta')
    plt.ylabel('sigma')
    plt.title(titles[var])
    plt.colorbar()
    plt.show()


def get_data(file):
    model = agent.set_model(file)
    net_last,ms_last,ang_vel = analyse.vortex_densities(model,model.tmax-1)
    peak_number, r_last, m_last = analyse.plot_m(model)
    analyse.get_velocity(model)
    last_mean,last_stdv = analyse.mean_velocity(model)
    analyse.pd_file(model, [net_last,ms_last,peak_number, r_last, m_last,last_mean,last_stdv])
    return
#model = todo(500,20,128,0.3,0.9,'grad',[-1,1],'q2D','flat')[1]
model = agent.set_model('data_q2D_p_0.6_1.5_0')
visualise.animate(model,False,'phase')
visualise.animate(model,False,'vorticity')
#model = agent.set_model('data_q2D_D_0_0.3_0')
#get_data('data_2D_D_0.4_0.5_0')
#print(model.theta.T[99][20] -model.theta.T[98][20])
#analyse.pd_file(model,[])
#file = np.array(pickle.load(open('vortices/pd_2D_grad_500.p', 'rb')))
#print(file)
#listoff()

#phase_diagram()
#print(model.omegas)
#css = analyse.css_t(model,499)
#plt.plot(np.array(css[0]).T[1])
#print(len(signal.find_peaks(np.array(css[0]).T[1])[0]))
#visualise.animate(model,False,'vorticity')
#plt.show()
#ctxy = analyse.cat_xy(model,10,10)
#plt.plot(ctxy)
#plt.show()

