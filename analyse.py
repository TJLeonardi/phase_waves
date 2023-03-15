import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft2, fft2
import visualise
import agent
import pickle
import os
from math import fabs, pi


def vortex_frame(model,frame):
    model.find_vorticity_2()
    values_before = model.vorticity[frame] #% (2 * np.pi)
    values = values_before.flatten()
    for i in range(len(values)):
        if values[i] > np.pi:
            values[i] -= 2 * np.pi

    return values_before.flatten()

def vortex_densities(model,frame):
    values = vortex_frame(model,frame)

    density = np.mean(values)
    squared_density = np.sum(values**2)/len(values)
    return density, squared_density

def vorsites(model,frame):
    values = vortex_frame(model,frame).reshape((model.N,model.M))
    mean = np.mean(np.sign(values)*values)
    sites = []
    for i in range(model.N):
        for j in range(1,model.M):
            if abs(values[i][j]) >= 0.1: #<= 4.5 * mean:
                #values[i][j] = 0

                sites.append([i,j])
    #plt.imshow(values, cmap='PiYG',interpolation='none')
    #plt.show()
    return sites

def interdistances(model,frame):
    sites = vorsites(model,frame)
    distances = []
    shortest_distances = []
    for i in range(len(sites)):
        i_distances = []
        for j in range(i):
            ydist = sites[i][1]-sites[j][1]
            if ydist > 10:
                ydist = 20 - ydist
            distance = np.sqrt((sites[i][0]-sites[j][0])**2 + (ydist)**2 )
            #if distance**2 > 2:
            i_distances.append(distance)
            distances.append(distance)
        if len(i_distances) > 1:
            shortest_distances.append(np.sort(i_distances)[0])
    distances.sort()
    mean = np.mean(distances)
    median = np.median(distances)
    #expected_mean = np.exp(1/model.eta)
    #plt.hist(distances,density=True)
    #plt.axvline(mean,color='r')
    #plt.axvline(median,color='g')

    #plt.axvline(expected_mean,color='k')

    #plt.xlabel('inter-vortex distance')
    #plt.title(r'distances at frame {}. green = median, red = mean, black = Lv'.format(frame))
    title = visualise.get_title(model,'interdistances')
    #plt.savefig('vortices/' + title + '.png')
    #plt.show()
    return shortest_distances,distances

def interdistances_multiframe(model,start, end):
    distances = []
    for t in range(start,end):
        distances.append(np.array(interdistances(model,t)[0]))
    distances = np.concatenate(np.array(distances).flatten())
    print(distances)
    mean = np.mean(distances)
    median = np.median(distances)

    plt.hist(distances, density=True,bins=20)
    plt.axvline(mean, color='r')
    plt.axvline(median, color='g')
    if model.eta > 0:
        expected_mean = np.exp(1 / model.eta)
        plt.axvline(expected_mean, color='k')

    plt.xlabel('inter-vortex distance')
    plt.title(r'distances between frames {} and {}. green = median, red = mean, black = Lv'.format(start,end))
    title = visualise.get_title(model, 'interdistances_multi')
    plt.savefig('vortices/' + title + '.png')
    plt.show()
    return mean

def plot_vd(model):
    plt.clf()
    y = []
    for t in range(model.tmax):
        y.append(vortex_densities(model,t))
    y = np.array(y)
    fig,axs = plt.subplots(2)
    axs[0].set_title(visualise.get_title(model,'vortex density in time'))
    axs[0].set_ylabel('net vorticity')
    #axs[0].set_xlabel('t')
    axs[0].plot(y.T[0])
    axs[0].axhline(y=0, color='r', linestyle='-')
    axs[1].plot(y.T[1])
    axs[1].set_ylabel('mean squared vorticity')
    axs[1].set_xlabel('t')
    plt.yscale('log')

    title = visualise.get_title(model, variable='vd', count=model.count)

    plt.savefig('vortices/' + title + '.png')
    plt.show()
    net_mean = np.mean(y.T[0][int(model.tmax/2):])
    ms_mean = np.mean(y.T[1][int(model.tmax / 2):])
    return net_mean,ms_mean

def magnetisation(model, frame):
    values_before = model.theta.T[frame] % (2 * np.pi)
    values = values_before.flatten()
    for i in range(len(values)):
        if values[i] > np.pi:
            values[i] -= 2 * np.pi
    M = np.mean(values)

    return M

def pd_file(model,vals='none'):
    title = r'pd_{}_{}'.format(model.dim,model.q2D)
    if title in os.listdir('vorticity'):
        pd_update(model,vals)
        return title
    else:
        data = [model.sigma,model.eta,vals[0],vals[1],vals[2]]
        pickle.dump(data, open('vortices/' + title + '.p', "wb"))
        return title


def pd_update(model,vals):
    title = r'pd_{}_{}'.format(model.dim,model.q2D)
    temp = pickle.load(open('vortices/' + title + '.p', 'rb'))
    temp.append([model.sigma,model.eta,vals[0],vals[1],vals[2]])
    pickle.dump(temp, open('vortices/' + title + '.p', "wb"))
    return

def plot_m(model):
    plt.clf()
    y = []
    for t in range(model.tmax):
        y.append(magnetisation(model, t))
    y = np.array(y)
    plt.plot(y)
    plt.title('magnetisation')
    plt.show()
    title = visualise.get_title(model, variable='magnetisation', count=model.count) + r'_t={}'.format(t)

    plt.savefig('vortices/' + title + '.png')


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
    title = visualise.get_title(model, variable='pcorr', count=model.count) + r'_t={}'.format(t)

    plt.savefig('correlations/' + title + '.png')

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
    title = visualise.get_title(model, variable='vcorr', count=model.count) + r'_t={}'.format(t)

    plt.savefig('correlations/' + title + '.png')
    return

def vdist_t(model,t):
    v_t = (model.theta.T[t + 1].reshape((model.N, model.M)) - model.theta.T[t].reshape((model.N, model.M))) % (
                2 * np.pi)
    return

def averagephase(filelist):
    models = []
    for i in range(len(filelist)):
        models.append(agent.set_model(filelist[i]))



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