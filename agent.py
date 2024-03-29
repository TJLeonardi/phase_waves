import numpy as np
from scipy.integrate import solve_ivp
import pickle
import os
from datetime import datetime
import visualise

epsilon = 1

class Kuramoto:
    def __init__(self, tmax, N, M=1, sigma=0.43, eta=0.04, bc='fix', grad=None, dim='Unspecified',init='rand',which_omegas=False, omegas=None):
        self.W = 0
        if grad is None:
            grad = [0, 0]
        self.N = N
        self.M = M
        self.tmax = tmax
        self.Dt = 1
        self.sigma = sigma
        # sigma 0.2, eta 0.2
        self.eta = eta
        self.title = ''
        self.init = init
        self.bc = bc
        self.grad = grad
        self.dim = dim
        self.theta = np.zeros(N * M)
        # self.theta[int(N/2)] = np.pi
        self.example = np.zeros([N, M])
        if tmax < 10000:
            self.v_x = np.zeros([tmax, N, M])
            self.v_y = np.zeros([tmax, N, M])
            self.vorticity = np.zeros([tmax, N, M])
            self.energy = np.zeros([tmax, N, M])
            self.divergence = np.zeros([tmax, N, M])

        if which_omegas==False:
            self.omegas = np.random.normal(0, sigma, [N, M])
        else:
            self.omegas  = omegas
        #self.omegas[int(N/2),int(M/2)] = 2.5
        #self.velocity = np.array([np.zeros(N * M) for t in range(tmax - 1)])

        # self.theta[0] = theta0


        if sigma == 0:
            if bc == 'fix' or bc == 'periodic' or grad == [0,0]:
                self.theta = np.random.random(N*M) * 2 * np.pi
                for j in range(0,M*N,M):
                    self.theta[j] = 0
                    if j != 0:
                        self.theta[j-1] = 0
                self.theta[0] = 0
                self.theta[N*M-1] = 0
        if init == 'rand':
            self.theta = np.random.random(N * M) * 2 * np.pi
        elif init == 'one_wind':
            self.theta = np.linspace(0,2*np.pi,N*M)
        elif init == 'vort':
            self.theta = np.zeros((N * M))  # np.pi*np.random.normal(size=(Lx*Ly))
            self.loc = self.grad#[16,0]
            self.theta += ((self.make_topo_defect(1, N, M, self.loc[0], M/2) + self.make_topo_defect(-1, N, M, N-self.loc[0], M/2)) % (2* np.pi)).flatten()
            self.theta += np.sign(self.loc[1]) * np.array([np.linspace(0,abs(self.loc[1])*2*np.pi,N) for i in range(M)]).flatten()
            #self.theta += ((self.make_topo_defect(1, N, M, self.loc[0], 3*M/4+self.loc[1]) + self.make_topo_defect(-1, N, M, N-self.loc[0], 3*M/4-self.loc[1])) % (2* np.pi)).flatten()

        elif init == 'flat':
            self.theta = np.zeros((N * M)) + 0.5
        # elif sigma == 0 and init == 'const':
        #    self.theta = [np.pi for i in range(N*M)]

    def K_phi(self, phi):
        return epsilon*(np.sin(phi) + self.eta * (1 - np.cos(phi)))

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
            # print(r'fixed at {}'.format(t))
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
        a1 = self.K_phi(j_plus - theta) # coupling to the left
        a2 = self.K_phi(j_minus - theta) # coupling to the right
        a3 = self.K_phi(i_plus - theta) # coupling above
        a4 = self.K_phi(i_minus - theta) # coupling below
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
            y[:, 0] = self.omegas[:, 0] + (self.K_phi(-self.grad[0]) + self.K_phi(theta[:, 1] - theta[:, 0])) + a3[:,0] + a4[:,0]
            y[:, -1] = self.omegas[:, -1] + (self.K_phi(self.grad[1]) + self.K_phi(theta[:, -2] - theta[:, -1])) + a3[:,-1] + a4[:,-1]
        elif self.bc == 'flat':
            y[:, 0] = self.omegas[:, 0] + (self.K_phi(-self.grad[0]) + self.K_phi(theta[:, 1] - theta[:, 0])) + a3[:,0] + a4[:,0]
            y[:, -1] = self.omegas[:, -1] + (self.K_phi(self.grad[1]) + self.K_phi(theta[:, -2] - theta[:, -1])) + a3[:,-1] + a4[:,-1]

            y[0, :] = self.omegas[0,:] + (self.K_phi(-self.grad[0]) + self.K_phi(theta[1,:] - theta[ 0,:])) + a1[0,:] + a2[0,:]
            y[-1, :] = self.omegas[ -1,:] + (self.K_phi(self.grad[1]) + self.K_phi(theta[ -2,:] - theta[ -1,:])) + a1[-1,:] + a2[-1,:]
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
            print('here')
            sol = solve_ivp(self.rhs_2d, [0, self.tmax], self.theta, method='LSODA',t_eval=range(self.tmax))
            print('gone')
        else:
            print('Error dim not present')
            return
        self.theta = sol.y
        return

    def save(self):
        title = visualise.get_title(self, 'data')
        date = datetime.today().strftime('%Y%m%d')
        files = os.listdir('data')
        files.sort()
        count = 0
        for file in files:
            if title == file[:-4]:
                count += 1
        title = r'{}_c{}_{}_{}_{}'.format(date, count, self.sigma, self.eta,self.grad)
        title = self.find_title()
        self.title = title

        stats = {'N': self.N, 'M': self.M, 'tmax': self.tmax, 'sigma': self.sigma,
                 'eta': self.eta, 'bc': self.bc, 'grad': self.grad, 'dim': self.dim, 'count':self.count,'W':self.W,'Dt':self.Dt}
        print(stats)
        print(title)
        data = {'stats': stats, 'omegas': self.omegas, 'theta': self.theta}
        pickle.dump(data, open('data/' + title + '.p', "wb"))
        return

    def diff(self):
        self.example = np.transpose(self.theta)[-2] - np.pi
        sign = np.sign(self.example - np.roll(self.example, -1))
        self.diffs = abs((self.example - np.roll(self.example, -1))) % (2 * np.pi)
        for i in range(self.N - 1):
            self.example[i:] = self.example[i:] + sign[i + 1] * self.diffs[i + 1]

    def pick(self, t):
        return np.transpose(self.theta)[t] % (2 * np.pi)

    def find_title(self):
        title = visualise.get_title(self,'data') #+ '_' + str(self.grad)
        chars = len(list(title))
        files = os.listdir('data')
        files.sort()
        count = 0
        for file in files:
            if file[:chars] == title:
                count += 1
        #title = title + '_' + str(count)
        self.count = count
        return title

    def find_vorticity_2(self,div=False):
        self.v_x = np.zeros([len(self.theta.T), self.N, self.M])
        self.v_y = np.zeros([len(self.theta.T), self.N, self.M])
        self.vorticity = np.zeros([len(self.theta.T), self.N, self.M])
        if div:
            self.find_divergence()
        for t in range(len(self.theta.T)):
            if not div:
                theta = np.reshape(self.theta.T[t], (self.N, self.M))
            else:
                theta = np.reshape(self.divergence[t], (self.N, self.M))
            self.v_x[t] = np.angle(np.exp(1j*(theta - np.roll(theta, 1, axis=1))))
            self.v_y[t] = np.angle(np.exp(1j*(theta - np.roll(theta, 1, axis=0))))
            if self.bc == 'grad':
                self.v_x[t][:, 0] = np.array([self.grad[0] for i in range(len(theta[:,0]))])
            if self.bc == 'flat':
                self.v_x[t][:, 0] = np.array([self.grad[0] for i in range(len(theta[:, 0]))])
                self.v_y[t][0,:] = np.array([self.grad[1] for i in range(len(theta[0,:]))])
            # self.v_y[t][0,:] = 0
            self.vorticity[t] = (self.v_y[t] - np.roll(self.v_y[t], 1, axis=1)) - (
                        self.v_x[t] - np.roll(self.v_x[t], 1, axis=0))
    def find_energy(self):
        for t in range(self.tmax):
            theta = np.reshape(self.theta.T[t], (self.N, self.M))
            self.v_x[t] = np.angle(np.exp(1j*(theta - np.roll(theta, 1, axis=1))))
            self.v_y[t] = np.angle(np.exp(1j*(theta - np.roll(theta, 1, axis=0))))
            self.energy[t][:,:] = self.v_x[t][:,:]**2 + self.v_y[t][:,:]**2

    def find_divergence(self):
        for t in range(self.tmax):
            theta = np.reshape(self.theta.T[t], (self.N, self.M))
            self.v_x[t] = np.angle(np.exp(1j*(theta - np.roll(theta, 1, axis=1))))
            self.v_y[t] = np.angle(np.exp(1j*(theta - np.roll(theta, 1, axis=0))))
            self.divergence[t] = self.v_x[t] - np.roll(self.v_x[t], 1, axis=1) +self.v_y[t] - np.roll(self.v_y[t], 1, axis=0)

    def make_topo_defect(self,k, Lx, Ly, x0, y0):
        x, y = np.meshgrid((np.arange(Lx) - x0), (np.arange(Ly) - y0))
        theta = np.arctan2(y, x)
        return (k * theta) % (2 * np.pi)


class cKPZ(Kuramoto):
    def K_phi(self,phi):
        return epsilon*(np.angle(np.exp(1j*phi)) + self.eta*(np.angle(np.exp(1j*phi))**2/2))

class temp_noise(Kuramoto):
    def __init__(self, tmax, N, M=1, sigma=0.43, eta=0.04, bc='fix', grad=None, dim='Unspecified',init='rand',which_omegas=False, omegas=None,W=0.,Dt=1):
        Kuramoto.__init__(self,tmax, N, M, sigma, eta, bc, grad, dim,init,which_omegas, omegas,)
        self.W = W
        self.Dt = Dt
        #self.velocity = np.zeros(N*M)
    def solve(self,dim=1):
        res = 100
        theta_init = self.theta
        self.theta = np.zeros([int(self.tmax), len(theta_init)])
        self.theta[0] = theta_init
        for t in range(self.tmax - 1):
            theta_t = np.reshape(self.theta[t], (self.N, self.M))
            j_plus = np.roll(theta_t, 1, axis=1)
            j_minus = np.roll(theta_t, -1, axis=1)
            i_plus = np.roll(theta_t, 1, axis=0)
            i_minus = np.roll(theta_t, -1, axis=0)
            a1 = self.K_phi(j_plus - theta_t)  # coupling to the left
            a2 = self.K_phi(j_minus - theta_t)  # coupling to the right
            a3 = self.K_phi(i_plus - theta_t)  # coupling above
            a4 = self.K_phi(i_minus - theta_t)  # coupling below
            w = np.random.normal(0, self.W, (self.N, self.M))
            y = (a1 + a2 + a3 + a4 + self.omegas) * self.Dt + np.sqrt(self.Dt) * w
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
                y[:, 0] = self.Dt * (self.omegas[:, 0] + (
                            self.K_phi(-self.grad[0]) + self.K_phi(theta_t[:, 1] - theta_t[:, 0])) + a3[:, 0] + a4[:,
                                                                                                                0])
                y[:, -1] = self.omegas[:, -1] + (
                        self.K_phi(self.grad[1]) + self.K_phi(theta_t[:, -2] - theta_t[:, -1])) + a3[:, -1] + a4[:, -1]
            elif self.bc == 'flat':
                y[:, 0] = (self.omegas[:, 0] + (self.K_phi(-self.grad[0]) + self.K_phi(theta_t[:, 1] - theta_t[:, 0]))
                           + a3[:, 0] + a4[:, 0]) * self.Dt
                y[:, -1] = (self.omegas[:, -1] + (
                            self.K_phi(self.grad[1]) + self.K_phi(theta_t[:, -2] - theta_t[:, -1]))
                            + a3[:, -1] + a4[:, -1]) * self.Dt
                y[0, :] = (self.omegas[0, :] + (self.K_phi(-self.grad[0]) + self.K_phi(theta_t[1, :] - theta_t[0, :]))
                           + a1[0, :] + a2[0, :]) * self.Dt
                y[-1, :] = (self.omegas[-1, :] + (self.K_phi(self.grad[1]) + self.K_phi(theta_t[-2, :]- theta_t[-1, :])) + a1[-1,:] + a2[-1,:]) * self.Dt
            self.theta[int(t) + 1] = self.theta[int(t)] + y.flatten()
        print(np.shape(self.theta))
        keep = np.zeros([int(self.tmax/res), len(theta_init)])
        for i in range(int(self.tmax/res)):
            keep[i] = self.theta[res*i]
        self.theta = keep.T
        return

class temp_noiseKPZ(cKPZ):
    def __init__(self, tmax, N, M=1, sigma=0.43, eta=0.04, bc='fix', grad=None, dim='Unspecified',init='rand',which_omegas=False, omegas=None,W=0.,Dt=1):
        Kuramoto.__init__(self,tmax, N, M, sigma, eta, bc, grad, dim,init,which_omegas, omegas,)
        self.W = W
        self.Dt = Dt
        #self.velocity = np.zeros(N*M)
    def solve(self,dim=1):
        res = 100
        theta_init = self.theta
        self.theta = np.zeros([int(self.tmax), len(theta_init)])
        self.theta[0] = theta_init
        for t in range(self.tmax - 1):
            theta_t = np.reshape(self.theta[t], (self.N, self.M))
            j_plus = np.roll(theta_t, 1, axis=1)
            j_minus = np.roll(theta_t, -1, axis=1)
            i_plus = np.roll(theta_t, 1, axis=0)
            i_minus = np.roll(theta_t, -1, axis=0)
            a1 = self.K_phi(j_plus - theta_t)  # coupling to the left
            a2 = self.K_phi(j_minus - theta_t)  # coupling to the right
            a3 = self.K_phi(i_plus - theta_t)  # coupling above
            a4 = self.K_phi(i_minus - theta_t)  # coupling below
            w = np.random.normal(0, self.W, (self.N, self.M))
            y = (a1 + a2 + a3 + a4 + self.omegas) * self.Dt + np.sqrt(self.Dt) * w
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
                y[:, 0] = self.Dt*(self.omegas[:, 0] + (self.K_phi(-self.grad[0]) + self.K_phi(theta_t[:, 1] - theta_t[:, 0])) + a3[:,0] + a4[ :,0])
                y[:, -1] = self.omegas[:, -1] + (
                            self.K_phi(self.grad[1]) + self.K_phi(theta_t[:, -2] - theta_t[:, -1])) + a3[:, -1] + a4[:, -1]
            elif self.bc == 'flat':
                y[:, 0] = (self.omegas[:, 0] + (self.K_phi(-self.grad[0]) + self.K_phi(theta_t[:, 1] - theta_t[:, 0]))
                           + a3[:, 0] + a4[:,0]) * self.Dt
                y[:, -1] = (self.omegas[:, -1] + (self.K_phi(self.grad[1]) + self.K_phi(theta_t[:, -2] - theta_t[:, -1]))
                            + a3[:, -1] + a4[:,-1]) * self.Dt
                y[0, :] = (self.omegas[0, :] + (self.K_phi(-self.grad[0]) + self.K_phi(theta_t[1, :] - theta_t[0, :]))
                           + a1[0, :] + a2[0,:]) * self.Dt
                y[-1, :] = (self.omegas[-1, :] + (self.K_phi(self.grad[1]) + self.K_phi(theta_t[-2, :]
                                                                    - theta_t[-1, :])) + a1[-1, :] + a2[-1,:]) * self.Dt

            self.theta[int(t) + 1] = self.theta[int(t)] + y.flatten()
        print(np.shape(self.theta))
        keep = np.zeros([int(self.tmax/res), len(theta_init)])
        for i in range(int(self.tmax/res)):
            keep[i] = self.theta[res*i]
        self.theta = keep.T
        #self.theta = self.theta.T
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


def setup():
    dirs = os.listdir()
    if not 'data' in dirs:
        os.mkdir('data')
    if not 'animations' in dirs:
        os.mkdir('animations')
    if not 'frames' in dirs:
        os.mkdir('frames')
    if not 'corrs' in dirs:
        os.mkdir('corrs')
    if not '2Dmeasures':
        os.mkdir('2Dmeasures')

def read(title):
    '''Opens specified data file containing simulation details and phase values'''
    return pickle.load(open('data/' + title +'.p', 'rb'))

def set_model(title):
    file = read(title)
    if title[:8] == 'temporal':
        uploaded = temp_noise(file['stats']['tmax'], file['stats']['N'], file['stats']['M'], file['stats']['sigma'],
                            file['stats']['eta'], file['stats']['bc'], file['stats']['grad'], file['stats']['dim'],W=file['stats']['W'],Dt=file['stats']['Dt'])
    else:
        uploaded = Kuramoto(file['stats']['tmax'], file['stats']['N'],file['stats']['M'],file['stats']['sigma'],
                        file['stats']['eta'],file['stats']['bc'],file['stats']['grad'],file['stats']['dim'])
    uploaded.omegas = file['omegas']
    uploaded.theta = file['theta']
    uploaded.count = file['stats']['count']
    uploaded.title = title
    print('Model set')
    return uploaded


class Repeats:
    def __init__(self,number, tmax,N,M,sigma,eta,bc,grad,dim):
        self.repeats = []
        self.number = number
        if grad is None:
            grad = [0, 0]
        self.N = N
        self.M = M
        self.tmax = tmax
        self.sigma = sigma
        self.eta = eta
        self.bc = bc
        self.grad = grad
        self.dim = dim
        self.avgtheta = np.zeros([tmax,N * M])
        self.omegas = np.random.normal(0, sigma, [N, M])

    def create(self):
        for i in range(self.number):
            repeat = Kuramoto(self.tmax,self.N,self.M,self.sigma,self.eta,self.bc,self.grad,self.dim)
            repeat.omegas = self.omegas
            repeat.solve(dim=1)
            self.repeats.append(repeat)
        return

    def averagetheta(self):
        for i in range(self.number):
                self.avgtheta += self.repeats[i].theta.T  # %(2*np.pi)
        self.avgtheta = self.avgtheta/self.number
        return


    def ploteach(self):
        for i in range(self.number):
            visualise.plot1D_frame(self.repeats[i], self.tmax-5)
        return


class Branch(Kuramoto):
    def __init__(self,tmax,N,M,sigma,eta,bc,grad,dim,init):
        super().__init__(tmax,N,M,sigma,eta,bc,grad,dim,init)
        self.theta = np.zeros(N * M * 2)
        self.circle_centre = [int(N / 2), int(3 * M / 5)]
        self.circumference, self.inside = self.find_circumference()
        self.omegas = np.random.normal(0, sigma, [2, N, M])
        for i in range(len(self.circumference)):
            self.omegas[0][self.circumference[i][0]][self.circumference[i][1]] = self.omegas[1][i][0]

    def rhs_branch(self,t,z0):
        theta, branch = z0.reshape(2, self.N * self.M)

        intermed_t = theta.reshape(self.N, self.M)
        intermed_b = branch.reshape(self.N, self.M)

        for i in range(len(self.circumference)):
            intermed_t[self.circumference[i][0]][self.circumference[i][1]] = intermed_b[i][0]

        for i in range(len(self.inside)):
            intermed_t[self.inside[i][0]][self.inside[i][1]] = 0

        j_plus = np.roll(intermed_t, 1, axis=1)
        j_minus = np.roll(intermed_t, -1, axis=1)
        i_plus = np.roll(intermed_t, 1, axis=0)
        i_minus = np.roll(intermed_t, -1, axis=0)
        for i in range(len(self.inside)):
            for frame in [j_plus,j_minus,i_plus,i_minus]:
                frame[self.inside[i][0]][self.inside[i][1]] = intermed_t[self.inside[i][0]][self.inside[i][1]]


        a1 = self.K_phi(j_plus - intermed_t)
        a2 = self.K_phi(j_minus - intermed_t)
        a3 = self.K_phi(i_plus - intermed_t)
        a4 = self.K_phi(i_minus - intermed_t)
        y1 = a1 + a2 + a3 + a4 + self.omegas[0]

        k_plus = np.roll(intermed_b, 1, axis=1)
        k_minus = np.roll(intermed_b, -1, axis=1)
        l_plus = np.roll(intermed_b, 1, axis=0)
        l_minus = np.roll(intermed_b, -1, axis=0)
        b1 = self.K_phi(k_plus - intermed_b)
        b2 = self.K_phi(k_minus - intermed_b)
        b3 = self.K_phi(l_plus - intermed_b)
        b4 = self.K_phi(l_minus - intermed_b)
        y2 = b1 + b2 + b3 + b4 + self.omegas[1]


        for i in range(len(self.inside)):
            y1[self.inside[i][0]][self.inside[i][1]] = 0

        for i in range(self.N):
            y2[i][0] = y1[self.circumference[i][0],self.circumference[i][1]] + b2[i][0]# + b3[i][0] + b4[i][0]
            y1[self.circumference[i][0], self.circumference[i][1]] = y2[i][0]
        if self.bc == 'fix':
            y1[:, 0] = 0
            y1[:, -1] = 0
            y2[:, -1] = 0
        elif self.bc == 'grad':
            y1[:, 0] = self.omegas[0][:, 0] + (self.K_phi(-self.grad[0]) + self.K_phi(intermed_t[:, 1] - intermed_t[:, 0]))
            y1[:, -1] = self.omegas[0][:, -1] + (self.K_phi(self.grad[1]) + self.K_phi(intermed_t[:, -2] - intermed_t[:, -1]))
            y2[:, -1] = self.omegas[1][:, -1] + (self.K_phi(self.grad[1]) + self.K_phi(intermed_b[:, -2] - intermed_b[:, -1]))

        ztheta = y1.flatten()
        zbranch = y2.flatten()
        output = np.array([ztheta, zbranch]).flatten()
        return output

    def solve_b(self):
        sol = solve_ivp(self.rhs_branch, [0, self.tmax], self.theta, method='LSODA', t_eval=range(self.tmax))
        self.theta = sol.y
        return

    def find_circumference(self):
        theta, branch = self.theta.reshape(2, self.N * self.M)

        manit = theta.reshape(self.N, self.M)
        manib = branch.reshape(self.N, self.M)
        count = 0
        r = self.N / (2 * np.pi)
        circumference = []
        inside = []
        tans = []
        manit[:][:] = 50
        centre = self.circle_centre# [int(self.N / 2), int(3 * self.M / 5)]
        print(centre)
        for i in range(self.N):
            for j in range(self.M):
                dist = np.sqrt((i - centre[0]) ** 2 + (j - centre[1]) ** 2)
                if dist < r + 1 and dist > r:
                    count += 1
                    manit[i][j] = 100
                    circumference.append(tuple([i, j]))
                    # print(dist)
                    if (j - centre[1]) == 0:
                        tans.append(-np.pi / 2)
                    else:
                        tans.append(np.arctan((i - centre[0]) / (j - centre[1])))

                if dist < r:
                    inside.append([i, j])
                    manit[i, j] = 0
        print(circumference)
        diction = dict(zip(circumference[:int(self.N / 2)], tans[:int(self.N / 2)]))
        diction2 = dict(zip(circumference[int(self.N / 2):], tans[int(self.N / 2):]))

        a1 = sorted(diction.items(), key=lambda item: item[1])
        a2 = sorted(diction2.items(), key=lambda item: item[1])
        b1 = np.array(a1[int(len(a1) / 2):], dtype=object).T[0]
        b2 = np.array(a1[:int(len(a1) / 2)], dtype=object).T[0]
        b3 = np.array(a2[int(len(a2) / 2):], dtype=object).T[0]
        b4 = np.array(a2[:int(len(a2) / 2)], dtype=object).T[0]
        res1 = [list(ele) for ele in b1]
        res2 = [list(ele) for ele in b2]
        res3 = [list(ele) for ele in b3]
        res4 = [list(ele) for ele in b4]

        circumference = np.concatenate([res1, res2, res3, res4])
        print(circumference)
        return circumference,inside





#centre = [int(N/2),int(M/2)]
 #           theta = np.reshape(self.theta, (self.N, self.M))
  #          for i in range(self.N):
   #             for j in range(self.M):
    #                if centre[0] < j < 3*centre[0]/2:
     #                   if centre[1] <= i < 3*centre[1]/2:
      #                      theta[i, j] = np.arctan((i-centre[1])/(j-centre[0]))
       #                 elif i > centre[0]/2:
        #                    theta[i, j] = np.arctan((i - centre[0]) / (j - centre[1]))
         #           elif centre[0] > j > centre[0]/2:
          #              if centre[1] <= i < 3*centre[0]/2:
           #                 theta[i, j] = np.arctan((i-centre[1])/(j-centre[0])) - np.pi
            #            elif i > centre[0]/2:
             #               theta[i, j] = np.arctan((i-centre[1])/(j-centre[0])) + np.pi
              #      else:
               #         if i >= centre[1]:
                #            theta[i, j] = np.pi/2
                 #       else:
                  #          theta[i, j] = -np.pi/2
        #def find_vorticity(self):
        #    self.v_x = np.zeros([self.tmax, self.N, self.M])
        #    self.v_y = np.zeros([self.tmax, self.N, self.M])
        #    self.vorticity = np.zeros([self.tmax, self.N, self.M])
        #    for t in range(self.tmax):
        #        theta = np.reshape(self.theta.T[t], (self.N, self.M))
        #        self.v_x[t] = theta - np.roll(theta, 1, axis=1)
        #        self.v_y[t] = theta - np.roll(theta, 1, axis=0)
        #        self.v_x[t][:, 0] = 0  # np.array([self.grad[0] for i in range(len(theta[:,0]))])
        #        # self.v_y[t][0,:] = 0
        #        self.vorticity[t] = (self.v_y[t] - np.roll(self.v_y[t], 1, axis=1)) - (
        #                    self.v_x[t] - np.roll(self.v_x[t], 1, axis=0))
                ##self.vorticity[t] = self.vorticity[t] - self.vorticity[0]
