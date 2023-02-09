import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as am
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import pickle
import os
from datetime import datetime
from scipy.fft import fft, ifft,fftfreq,ifft2,fft2,rfft2,irfft2


class Kuramoto:
    def __init__(self, theta0, tmax, N, M=1, sigma=0.43, eta=0.04, bc='fix', grad=[0, 0], init='rand'):
        self.N = N
        self.M = M
        self.tmax = tmax
        self.sigma = sigma
        # sigma 0.2, eta 0.2
        self.eta = eta
        self.title = ''
        self.bc = bc
        self.grad = grad
        self.dims = ''
        self.theta = np.zeros(N * M)
        #self.theta[int(N/2)] = np.pi
        self.example = np.zeros([N, M])
        self.v_x = np.zeros([tmax,N,M])
        self.v_y = np.zeros([tmax,N,M])
        self.vorticity = np.zeros([tmax,N,M])


        # self.theta[0] = theta0
        self.omegas = np.random.normal(0, sigma, [N, M])

        if sigma == 0 and init=='rand' :
            if bc == 'fix' or bc == 'periodic' or grad == [0,0]:

                self.theta = np.random.random(N*M) * 2 * np.pi
                for j in range(0,M*N,M):
                    self.theta[j] = 0
                    if j != 0:
                        self.theta[j-1] = 0
                self.theta[0] = 0
                self.theta[N*M-1] = 0
        #elif sigma == 0 and init == 'const':
        #    self.theta = [np.pi for i in range(N*M)]

    def K_phi(self, phi):
        return np.sin(phi) + self.eta * (1 - np.cos(phi))

    def rhs(self, t, theta):
        # print(f'theta is {theta}')
        # omegas = np.random.normal(0,sigma,self.N)
        psi_0 = np.roll(theta, -1)
        psi_1 = theta
        psi_2 = np.roll(theta, 1)
        a1 = self.K_phi(psi_2 - psi_1)
        a2 = self.K_phi(psi_0 - psi_1)
        y = (a1 + a2 + self.omegas)
        # fixed boundary conditions
        if self.bc == 'fix':
            y[0] = 0
            y[-1] = 0
            #print(r'fixed at {}'.format(t))
        elif self.bc == 'grad':
            y[0] = self.omegas[0] + (self.K_phi(-self.grad[0]) + self.K_phi(theta[1] - theta[0]))
            y[-1] = self.omegas[-1] + (self.K_phi(self.grad[1]) + self.K_phi(theta[-2] - theta[-1]))

        return y

    def rhs_2d(self, t, z0):
        theta = np.reshape(z0, (self.N, self.M))
        # theta = np.zeros([self.N,self.M])
        # theta = theta.flatten()
        # for i in range(int(len(z0)/self.N)):
        #    theta[:,i] = z0[(self.N)*(i):(self.N)*(i+1)]
        j_plus = np.roll(theta, 1, axis=1)
        j_minus = np.roll(theta, -1, axis=1)
        i_plus = np.roll(theta, 1, axis=0)
        i_minus = np.roll(theta, -1, axis=0)
        a1 = self.K_phi(j_plus - theta)
        a2 = self.K_phi(j_minus - theta)
        a3 = self.K_phi(i_plus - theta)
        a4 = self.K_phi(i_minus - theta)
        y = a1 + a2 + a3 + a4 + self.omegas
        if self.bc == 'all_fix':
            y[0, :] = 0
            y[:, 0] = 0
            y[-1, :] = 0
            y[:, -1] = 0
        elif self.bc == 'fix':
            # y[0,:] =  y[-1,:]
            y[:, 0] = 0
            y[:, -1] = 0
        elif self.bc == 'grad':
            y[:,0] = self.omegas[:,0] + (self.K_phi(-self.grad[0]) + self.K_phi(theta[:,1] - theta[:,0]))
            y[:,-1] = self.omegas[:,-1] + (self.K_phi(self.grad[1]) + self.K_phi(theta[:,-2] - theta[:,-1]))
        elif self.bc == 'custom':
            y[:, 0] = 0
            y[:, -1] = 0
        elif self.bc == 'periodic':
            pass

        # reshape, flatten
        # z= np.concatenate(y)
        z = y.flatten()
        return z

    def solve(self, dim=1):
        if dim == 1:
            self.omegas = self.omegas.flatten()
            sol = solve_ivp(self.rhs, [0, self.tmax], self.theta, method='LSODA',t_eval=range(self.tmax))
        elif dim == 2:
            sol = solve_ivp(self.rhs_2d, [0, self.tmax], self.theta, method='LSODA',t_eval=range(self.tmax))

        self.theta = sol.y
        return

    def save(self):
        stats = {'N':self.N, 'M':self.M, 'tmax':self.tmax,'sigma':self.sigma,
                 'eta':self.eta, 'bc':self.bc,'grad':self.grad}
        print(stats)
        data = {'stats':stats, 'omegas':self.omegas, 'theta':self.theta}
        date = datetime.today().strftime('%Y%m%d')
        files = os.listdir('data')
        files.sort()
        count = 0
        for file in files:
            if file[:8] == date:
                count += 1
        title = r'{}_c{}_{}_{}'.format(date,count,self.sigma,self.eta)
        self.title = title
        pickle.dump(data, open('data/' + title +'.p', "wb"))
        return

    def diff(self):
        self.example = np.transpose(self.theta)[-2] - np.pi
        sign = np.sign(self.example - np.roll(self.example, -1))
        self.diffs = abs((self.example - np.roll(self.example, -1))) % (2 * np.pi)
        for i in range(self.N - 1):
            self.example[i:] = self.example[i:] + sign[i + 1] * self.diffs[i + 1]

    def pick(self, t):
        return np.transpose(self.theta)[t] % (2 * np.pi)

    def vorticity(self):
        for t in range(self.tmax):
            theta = self.theta.T[t]
            self.v_x[t] = theta - np.roll(theta, 1, axis=1)
            self.v_y[t] = theta - np.roll(theta, 1, axis=0)
            self.vorticity[t] = (self.v_y[t] - np.roll(self.v_y[t],1,axis=1)) - (self.v_x[t] - np.roll(self.v_x[t],1,axis=1))

    def omega(self, t):
        theta = self.pick(t)
        shift(theta)
        for i in range(len(self.v)):
            self.v[i] = self.eta / self.N * (
                np.sum(1 - np.cos(np.roll(theta, 1) - theta) + 1 - np.cos(np.roll(theta, -1) - theta)))
            self.v[i] += np.mean(self.omegas)
            # wy
            self.v[i] += theta[i]
        return




def shift(phases, tol=1):
    z = np.exp(phases*1j)
    diff_z = np.abs(z[1:]-z[:-1])
    diff_ang = phases[1:]-phases[:-1]
    for (i, (dz, dtheta)) in enumerate(zip(diff_z, diff_ang)):
        if np.abs(dtheta) - dz > tol:
            if dtheta < 0:
                phases[i+1:] += np.pi*2
            else:
                phases[i+1:] -= np.pi*2

def plot1D_frame(model,t,save=True):
    fig,axs = plt.subplots(1,2)
    fig.set_figheight(7)
    fig.set_figwidth(15)

    phases = model.pick(t)
    shift(phases)
    axs[0].plot(phases)
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$\Theta$')
    axs[0].set_title(r'spatial phase distribution $t={}$, $\sigma={}, \eta ={}, bc ={}$'.format(t,model.sigma, model.eta,model.grad))

    im1 = axs[1].imshow(model.theta.T %(2*np.pi), cmap='twilight',aspect='auto',interpolation='none')
    #plt.colorbar(im1,orientation='vertical')
    cbar = fig.colorbar(im1, ticks=[0, np.pi,2*np.pi], orientation='vertical')
    cbar.set_ticklabels([r'$0$', r'$\pi$',r'$2\pi$'])
    axs[1].set_xlabel('$x$')
    axs[1].set_ylabel('$t$')
    axs[1].set_title('space time graph of phase evolution $\sigma={}, \eta ={}$'.format(model.sigma, model.eta))

    #plt.legend()
    if model.grad == [1, 1]:
        val = 'A'
    elif model.grad == [-1, 1]:
        val = 'B'
    elif model.grad == [1, -1]:
        val = 'C'
    elif model.grad == [0, 0] and model.bc == 'grad':
        val = 'D'
    elif model.bc == 'fix':
        val = 'fix'
    elif model.bc == 'periodic':
        val = 'p'
    title = r'1D_bc{}_{}_{}_{}'.format(val, t, model.sigma, model.eta)
    if save:
        plt.savefig(title + '.png')
    plt.show()

def plot2D_frame(t,model,save=True):
    im2 = plt.imshow(model.theta.T[t].reshape((model.N,model.M))% (2*np.pi), cmap='twilight',vmin=0, vmax=2*np.pi,interpolation='none')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(r'$t={}$, $\sigma={}, \eta ={}, bc = {}$'.format(t,model.sigma, model.eta,model.grad))
    #plt.colorbar(im2,orientation= "horizontal")
    cbar = plt.colorbar(im2, ticks=[0, np.pi, 2 * np.pi], orientation='horizontal')
    cbar.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    if model.grad == [1,1]:
        val = 'A'
    elif model.grad == [-1,1]:
        val = 'B'
    elif model.grad == [1,-1]:
        val = 'C'
    elif model.grad == [0, 0] and model.bc == 'grad':
        val = 'D'
    elif model.bc == 'fix':
        val = 'fix'
    elif model.bc == 'periodic':
        val = 'p'
    title = r'2D_bc{}_{}_{}_{}'.format(val, t, model.sigma, model.eta)
    #title = r'2D_tse_{}_{}_{}'.format(t,model.sigma, model.eta)
    if save:
        plt.savefig(title + '.png')
    plt.show()

def plot2D_frame_compare(t,model,save=True):
    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot2grid((2, 4), (0, 1), colspan=3)
    ax2 = plt.subplot2grid((2, 4), (1, 0))
    ax3 = plt.subplot2grid((2, 4), (1, 1), colspan=3)

    im2 = ax2.imshow(model.theta.T[t].reshape((20, 128)) % (2 * np.pi), cmap='twilight', vmin=0, vmax=2 * np.pi,
                     interpolation='none')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_title(r'$t={}$, $\sigma={}, \eta ={}$'.format(t, model.sigma, model.eta))
    divider2 = make_axes_locatable(ax2)
    # Append axes to the right of ax3, with 20% width of ax3
    cax3 = divider2.append_axes("right", size="20%", pad=0.05)
    cb3 = plt.colorbar(im2, cax=cax3, orientation="horizontal")
    if model.grad == [1,1]:
        val = 'A'
    elif model.grad == [-1,1]:
        val = 'B'
    elif model.grad == [1,-1]:
        val = 'C'
    elif model.grad == [0, 0] and model.bc == 'grad':
        val = 'D'
    elif model.bc == 'fix':
        val = 'fix'
    elif model.bc == 'periodic':
        val = 'p'
    title = r'2D_{}_{}_{}_{}'.format(val, t, model.sigma, model.eta)
    if save:
        plt.savefig(title + '.png')
    plt.show()


def animate(model,rem):
    fig = plt.figure()
    #print(model.theta.T[].reshape((20,128))% (2*np.pi))
    a= model.theta.T[0].reshape((model.N, model.M)) % (2 * np.pi)
    im = plt.imshow(a, cmap='twilight', vmin=0, vmax=2*np.pi, interpolation='none')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.colorbar(im, orientation="horizontal")

    def update(t):
        im.set_array(model.theta.T[t].reshape((model.N,model.M)) % (2*np.pi))
        plt.title(r'$t={}$, $\sigma={}, \eta ={}$'.format(t, model.sigma, model.eta))
        return [im]
    anim = am.FuncAnimation(fig, update,frames=[i for i in range(0,model.tmax)],repeat=False)
    if model.grad == [1,1]:
        val = 'A'
    elif model.grad == [-1,1]:
        val = 'B'
    elif model.grad == [1,-1]:
        val = 'C'
    elif model.grad == [0, 0] and model.bc == 'grad':
        val = 'D'
    elif model.bc == 'fix':
        val = 'fix'
    elif model.bc == 'periodic':
        val = 'p'
    else:
        val = 'custom'
    title = r'2Danim_{}_{}_{}'.format(val, model.sigma, model.eta)
    if rem:
        title = title + '_alt'
    anim.save(title +'.mp4',fps=25,extra_args=['-vcodec','libx264'])

def read(title):
    return pickle.load(open('data/' + title +'.p','rb'))

def set_model(title):
    file = read(title)
    uploaded = Kuramoto(0,file['stats']['tmax'], file['stats']['N'],file['stats']['M'],file['stats']['sigma'],
                        file['stats']['eta'],file['stats']['bc'],file['stats']['grad'])
    uploaded.omegas = file['omegas']
    uploaded.theta = file['theta']
    uploaded.title = title
    return uploaded


def todo(tmax,sigma,eta,bc,grad,shape):
    t_max = tmax
    length = 128
    #length= 64
    if shape =='2D':
        height = 128
        model = Kuramoto(0, t_max, height, length, sigma, eta, bc, grad)
        model.solve(dim=2)
        print('2D solved')
        model.save()
        plot2D_frame(model.tmax - 2, model)
        print('2D saved')
    elif shape == 'q2D':
        height = 20
        model = Kuramoto(0, t_max, height, length, sigma, eta, bc, grad)
        model.solve(dim=2)
        print('2D solved')
        model.save()
        plot2D_frame(model.tmax - 2, model)
        print('2D saved')
    elif shape == '1D':
        model = Kuramoto(0,t_max,length,1,sigma,eta,bc,grad)
        model.solve()
        print('1D solved')
        model.save()
        plot1D_frame(model,t = -2)
        print('1D saved')
    return model.title


def remove_harmonic(filename):
    db = read(filename)
    model = set_model(filename)
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
        #new_Z[:5][:40] = np.real(Z[:5][:40])

        new_yframe = ifft2(new_Z)
        print(new_yframe.shape)
        new_theta.append(new_yframe)
        print(np.array(new_theta).shape)
    new_theta = np.array(np.real(new_theta)).reshape(model.N * model.M, model.tmax)
    model.theta = new_theta
    animate(model,rem=True)


def anim2D(tmax,sigma,eta,bc,grad,shape):
    if shape == '2D':
        N = 128
    elif shape == 'q2D':
        N = 20
    eg2D = Kuramoto(0,tmax,N,128,sigma,eta,bc,grad)
    eg2D.solve(dim=2)
    animate(eg2D,False)


def anim_read(title):
    model = set_model(title)
    animate(model,False)
    return

def custom():
    custom = Kuramoto(0, 1000, 20, 128, 0.1, 0.5, 'custom', [0, 0])
    custom.omegas[10][64] = 2
    custom.solve(2)
    custom.save()
    animate(custom, False)


tmax = 1000
vals = [0,0.05,0.1,0.2,1]

bcs = ['fix','grad','grad','grad','grad','periodic']
grad = [[0,0],[1,1],[-1,1],[1,-1],[0,0],[]]
dims = ['1D','q2D','2D']


for i in [0.5]:
    for j in [1]:
        for k in [0]:
            for l in [1]:
                #title = todo(tmax,i,j,bcs[k],grad[k],dims[l])
                #anim_read(title)
                pass

def custom():
    custom = Kuramoto(0,1000,20,128,0.1,0.5,'custom',[0,0])
    custom.omegas[10][64] = 2
    custom.solve(2)
    custom.save()
    animate(custom,False)

    return

#remove_harmonic('20230208_c0_0.5_1')




#im3 = plt.imshow(np.real(new_yframe),interpolation='none')
#plt.show()


#anim2D(2000,0,0,'grad',[1,1])







#eg1D = Kuramoto(0,100,128,1,0,0.1,'fix',[0,0],'rand')
#eg1D.solve(dim=1)
#plot1D_frame(eg1D,-2,False)
#plt.show()



#todo(1,0.01,'fix',[1,1])

