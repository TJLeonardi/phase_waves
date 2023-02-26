import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft2, fft2
import visualise
import agent

def remove_harmonic(db,model):
    full = model.theta.reshape(model.N, model.M, model.tmax)
    # im2 = plt.imshow(full[-2] % (2*np.pi) , cmap='twilight',  vmin=0, vmax=2 * np.pi,
    #                interpolation='none')
    full = full % (2 * np.pi)
    new_theta = []
    for i in range(len(full)):
        yframe = full[i]
        Z = fft2(yframe)
        new_Z = np.zeros(Z.shape)
        print(yframe.shape)
        new_Z[3:][90:] = np.real(Z[3:][90:])
        # new_Z[5:10][:] = np.real(Z[5:10][:])
        # new_Z[:][40:50] = np.real(Z[:][40:50])
        # new_Z[:5][:40] = np.real(Z[:5][:40])

        new_yframe = ifft2(new_Z)
        print(new_yframe.shape)
        new_theta.append(new_yframe)
        print(np.array(new_theta).shape)
    new_theta = np.array(np.real(new_theta)).reshape(model.N * model.M, model.tmax)
    model.theta = new_theta
    visualise.animate(model, rem=True)

def vortex_densities(model,frame):
    model.find_vorticity()
    values = model.vorticity[frame].flatten() % (2 * np.pi)
    density = np.mean(values)
    squared_density = sum(i*i for i in values)/len(values)
    return density, squared_density

def plot_vd(model):
    y = []
    for t in range(model.tmax):
        y.append(vortex_densities(model,t))
    y = np.array(y)
    fig,axs = plt.subplots(2)
    axs[0].plot(y.T[0])
    axs[1].plot(y.T[1])
    plt.show()