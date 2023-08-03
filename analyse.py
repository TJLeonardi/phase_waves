import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft2, fft2
import visualise
import agent
import pickle
import os
from scipy import signal
from math import fabs, pi


def vortex_frame(model,frame):
    model.find_vorticity_2()
    values_before = model.vorticity[frame]  # % (2 * np.pi)
    values = values_before.flatten()
    # for i in range(len(values)):
    #    if values[i] > np.pi:
    #        values[i] -= 2 * np.pi

    return values

def vortex_densities(model,frame):
    values = vortex_frame(model,frame)
    sites = vorsites(model,frame)
    density = np.mean(values)
    squared_density = np.sum(values**2)/len(values)
    ang_vel =  np.sum(abs(values)/len(sites))/(2*np.pi)
    #print(len(values))
    #print(len(sites))
    return density, squared_density, ang_vel

def vorsites(model,frame,threshold=0.01):
    values = vortex_frame(model,frame).reshape((model.N,model.M))
    sites = []
    for i in range(model.N):
        for j in range(1,model.M):
            if abs(values[i][j]) >= threshold: #<= 4.5 * mean:
                #values[i][j] = 0
                sites.append([frame,i,j,np.sign(np.divide(values[i][j],2*np.pi))])
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
    fig,axs = plt.subplots(3)
    axs[0].set_title(visualise.get_title(model,'vortex density in time'))
    axs[0].set_ylabel('net vorticity')
    #axs[0].set_xlabel('t')
    axs[0].plot(y.T[0])
    axs[0].axhline(y=0, color='r', linestyle='-')
    axs[1].plot(y.T[1])
    axs[1].set_ylabel('mean squared vorticity')
    axs[1].set_xlabel('t')
    plt.yscale('log')
    axs[2].plot(y.T[2])
    axs[2].set_ylabel('mean angular velocity')
    axs[2].set_xlabel('t')


    title = visualise.get_title(model, variable='vd', count=model.count)

    plt.savefig('vortices/' + title + '.png')
    plt.show()
    #net_mean = np.mean(y.T[0][int(model.tmax/2):])
    #ms_mean = np.mean(y.T[1][int(model.tmax / 2):])
    net_last = y.T[0][-1]
    ms_last = y.T[1][-1]
    return net_last,ms_last

def rij(i_1,j_1,i_2,j_2):
    return np.sqrt(np.subtract(i_1,i_2)**2 + np.subtract(j_1,j_2)**2)

def delta(i_1,j_1,i_2,j_2,r):
    if rij(i_1,j_1,i_2,j_2) <= (r+1) and rij(i_1,j_1,i_2,j_2) > r :
            d = 1
    else:
            d=0
    return d


def css_t(model, t):
    csst = []
    csst_norm = []
    theta = model.theta.T[t].reshape((model.N,model.M))% (2*np.pi)
    if model.N > model.M:
        max = model.N
    else:
        max = model.M
    for r in range(0, max):
        cssr = 0
        Z = 0
        for i_1 in range(0,model.N,8):
            for j_1 in range(0,model.M,8):
                for i_2 in range(model.N):
                    for j_2 in range(model.M):
                        d = delta(i_1,j_1,i_2,j_2,r)
                        if d == 1:
                            shifted_theta = np.roll(np.roll(theta,i_2,axis=0),j_2,axis=1)
                            cssr += np.cos(theta[i_1,j_1] - shifted_theta[i_1,j_1]) * d
                            Z += d
        if Z == 0:
            csst_norm.append([(r + 1 / 2), 0])
            csst.append([(r + 1 / 2), 0])
        else:
            csst_norm.append([(r + 1 / 2) , np.divide(cssr, Z)])
            csst.append([(r + 1 / 2) , cssr])

    #print(f'{csst=}')
    return csst_norm, csst


def corr_auto_xy_t0(model,x,y,t_0=0):
    caxy_t0 = np.zeros(model.tmax)
    compare = model.theta.T[t_0].reshape((model.N, model.M))[x,y]
    for t in range(model.tmax):
        theta = model.theta.T[t].reshape((model.N, model.M)) # % (2 * np.pi)
        caxy_t0[t] = (theta[x,y] - compare)
    return caxy_t0


def corr_auto_xy(model,x,y):
    caxy = np.array([np.zeros(model.tmax) for t in range(model.tmax)])
    for t0 in range(model.tmax):
        caxy[t0] = corr_auto_xy_t0(model,x,y,t0)

def magnetisation(model, frame):
    values_before = model.theta.T[frame] % (2 * np.pi)
    values = values_before.flatten()
    for i in range(len(values)):
        if values[i] > np.pi:
            values[i] -= 2 * np.pi
    M = np.mean(values)
    E = np.mean(np.exp(1j*values))
    R = abs(E)
    T = np.angle(E)
    return M,R,T

def plot_m(model):
    plt.clf()
    m,r,t = np.array([magnetisation(model,t) for t in range(model.tmax)]).T
    fig,axs = plt.subplots(3)
    axs[0].plot(m)
    axs[0].set_title(visualise.get_title(model,'magnestisation',model.count))
    axs[0].set_ylabel('mean $\Theta$')
    axs[1].plot(r)
    axs[1].set_ylabel('abs mean exp(i$\Theta$)')
    axs[2].plot(t)
    axs[2].set_ylabel('arg mean exp(i$\Theta$)')
    axs[2].set_xlabel('t')
    peaks = signal.find_peaks(t)
    peak_number = len(peaks[0])*5
    r_last = (r[-1])
    m_last = m[-1]
    plt.show()
    #title = visualise.get_title(model, variable='magnetisation', count=model.count) + r'_t={}'.format(t)

    #plt.savefig('vortices/' + title + '.png')
    return peak_number,r_last,m_last

def get_velocity(model):
    #velocity = np.array([np.zeros(model.N*model.M) for t in range(model.tmax - 1)])
    for t in range(model.tmax-1):
        model.velocity[t] = np.subtract(model.theta.T[t+1] , model.theta.T[t] )
    model.velocity = model.velocity.reshape(model.tmax-1,model.N,model.M)
    return

def mean_velocity(model):
    means= np.mean(np.mean(abs(model.velocity),axis=1),axis=1)
    stdvs = np.std(model.velocity.reshape(model.tmax-1,model.N*model.M),axis=1)
    fig,axs = plt.subplots(2)
    axs[0].plot(means)

    axs[0].set_title(visualise.get_title(model,'mean and stdv of velocity',model.count))
    axs[0].set_ylabel('mean velocity')
    axs[1].plot(stdvs)
    axs[1].set_ylabel('stdev velocity')
    axs[1].set_xlabel('t')
    last_mean = means[-1]
    last_stdv = stdvs[-1]
    plt.show()
    return last_mean,last_stdv


def pd_file(model,vals='none'):
    title = r'pd_{}_{}_{}.p'.format(model.dim,model.bc,model.tmax)
    if title in os.listdir('vortices'):
        pd_update(model,vals)
        return title
    else:
        # data = [model.sigma,model.eta,vals[0],vals[1],vals[2]]
        new_row = [model.sigma, model.eta]
        for i in range(len(vals)):
            new_row.append(vals[i])

        pickle.dump([new_row], open('vortices/' + title, "wb"))
        return title


def pd_update(model,vals):
    title = r'pd_{}_{}_{}'.format(model.dim,model.bc,model.tmax)
    temp = pickle.load(open('vortices/' + title + '.p', 'rb'))
    new_row = [model.sigma,model.eta]
    for i in range(len(vals)):
        new_row.append(vals[i])
    temp.append(new_row)
    pickle.dump(temp, open('vortices/' + title + '.p', "wb"))
    return




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
    title = visualise.get_title(model, variable='pcorr', count=model.count) #+ r'_t={}'.format(t)

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