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

def pcorr_t(model,t):
    phases = model.theta.T[t].reshape((model.N,model.M)) % (2*np.pi)
    if model.dim == '1D':
        corr_t = np.zeros(len(phases))
        for r in range(len(phases)):
            diffs = 0
            for i in range(len(phases) - r):
                diffs += ((phases[i] - phases[i + r]) % (2 * np.pi))
                corr_r = diffs / len(phases)
            corr_t[r] = corr_r
        return np.array(corr_t)


def pcorr(model):
    corrs = np.array([np.zeros(model.N) for i in range(model.tmax - 1)])
    for t in range(model.tmax - 1):
        corrs[t] = pcorr_t(model, t)
    print(corrs.shape)
    corr_r = np.sum(corrs, axis=0) / model.tmax
    return corr_r


def plotpcorr(model):
    y = pcorr(model)
    plt.plot(y)
    plt.title('p')
    plt.show()

def vcorr_t(model,t):
    v_t = (model.theta.T[t+1].reshape((model.N,model.M))- model.theta.T[t].reshape((model.N,model.M))) #% (2*np.pi)
    if model.dim == '1D':
        corr_t = np.zeros(len(v_t))
        for r in range(len(v_t)):
            diffs = 0
            for i in range(len(v_t)-r):
                diffs += ((v_t[i]-v_t[i+r])) #% (2*np.pi))
            corr_r = diffs/len(v_t)
            corr_t[r] = corr_r
    else:
        return
    if t == model.tmax -5:
        plt.plot(corr_t)
        plt.title(t)
        plt.show()
    return np.array(corr_t)


def vcorr(model):
    corrs =np.array([np.zeros(model.N) for i in range(model.tmax-1)])
    for t in range(model.tmax-1):
        corrs[t] = vcorr_t(model,t)
    print(corrs.shape)
    corr_r = np.sum(corrs,axis=0)/model.tmax
    return corr_r

def plotvcorr(model):
    y = vcorr(model)
    plt.plot(y)
    plt.title('v')
    plt.show()
    return

def vdist_t(model,t):
    v_t = (model.theta.T[t + 1].reshape((model.N, model.M)) - model.theta.T[t].reshape((model.N, model.M))) % (
                2 * np.pi)
    return

def averagephase(filelist):
    models = []
    for i in range(len(filelist)):
        models.append(agent.set_model(filelist[i]))
