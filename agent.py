import numpy as np
from scipy.integrate import solve_ivp
import pickle
import os
from datetime import datetime


class Kuramoto:
    def __init__(self, tmax, N, M=1, sigma=0.43, eta=0.04, bc='fix', grad=None, dim='q2D',init='rand'):
        if grad is None:
            grad = [0, 0]
        self.N = N
        self.M = M
        self.tmax = tmax
        self.sigma = sigma
        # sigma 0.2, eta 0.2
        self.eta = eta
        self.title = ''
        self.bc = bc
        self.grad = grad
        self.dim = dim
        self.theta = np.zeros(N * M)
        # self.theta[int(N/2)] = np.pi
        self.example = np.zeros([N, M])
        self.v_x = np.zeros([tmax, N, M])
        self.v_y = np.zeros([tmax, N, M])
        self.vorticity = np.zeros([tmax, N, M])
        self.omegas = np.random.normal(0, sigma, [N, M])
        if dim == 'branch':
            self.theta = np.zeros(N*M*2)
            self.circle_centre = [int(N/2),int(3*M/5)]
            self.circumference,self.inside = self.find_circumference()
            self.omegas = np.random.normal(0, sigma, [2,N, M])
            for i in range(len(self.circumference)):
                self.omegas[0][self.circumference[i][0]][self.circumference[i][1]] = self.omegas[1][i][0]



        # self.theta[0] = theta0


        if sigma == 0 and init=='rand' :
            if bc == 'fix' or bc == 'periodic' or grad == [0,0]:

                self.theta = np.random.random(N*M) * 2 * np.pi
                for j in range(0,M*N,M):
                    self.theta[j] = 0
                    if j != 0:
                        self.theta[j-1] = 0
                self.theta[0] = 0
                self.theta[N*M-1] = 0
        # elif sigma == 0 and init == 'const':
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
            y[:, 0] = self.omegas[:, 0] + (self.K_phi(-self.grad[0]) + self.K_phi(theta[:, 1] - theta[:, 0]))
            y[:, -1] = self.omegas[:, -1] + (self.K_phi(self.grad[1]) + self.K_phi(theta[:, -2] - theta[:, -1]))
        elif self.bc == 'custom':
            y[:, 0] = 0
            y[:, -1] = 0
        elif self.bc == 'periodic':
            pass

        # reshape, flatten
        # z= np.concatenate(y)
        z = y.flatten()
        return z

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


    def solve(self, dim=1):
        if dim == 1:
            self.omegas = self.omegas.flatten()
            sol = solve_ivp(self.rhs, [0, self.tmax], self.theta, method='LSODA',t_eval=range(self.tmax))
        elif dim == 2:
            sol = solve_ivp(self.rhs_2d, [0, self.tmax], self.theta, method='LSODA',t_eval=range(self.tmax))
        elif dim == 3:
            sol = solve_ivp(self.rhs_branch, [0, self.tmax], self.theta, method='LSODA', t_eval=range(self.tmax))
        else:
            print('Error dim not present')
            return
        self.theta = sol.y
        return

    def save(self):
        stats = {'N': self.N, 'M': self.M, 'tmax': self.tmax, 'sigma': self.sigma,
                 'eta': self.eta, 'bc': self.bc, 'grad': self.grad, 'dim':self.dim}
        print(stats)
        data = {'stats': stats, 'omegas': self.omegas, 'theta': self.theta}
        date = datetime.today().strftime('%Y%m%d')
        files = os.listdir('data')
        files.sort()
        count = 0
        for file in files:
            if file[:8] == date:
                count += 1
        title = r'{}_c{}_{}_{}'.format(date, count, self.sigma, self.eta)
        self.title = title
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

    def find_vorticity(self):
        for t in range(self.tmax):
            theta = np.reshape(self.theta.T[t], (self.N, self.M))
            self.v_x[t] = theta - np.roll(theta, 1, axis=1)
            self.v_y[t] = theta - np.roll(theta, 1, axis=0)
            self.vorticity[t] = (self.v_y[t] - np.roll(self.v_y[t],1,axis=1)) - (self.v_x[t] - np.roll(self.v_x[t],1,axis=1))
            #self.vorticity[t] = self.vorticity[t] - self.vorticity[0]
   # def omega(self, t):
   #     theta = self.pick(t)
   #     shift(theta)
   #     for i in range(len(self.v)):
   #         self.v[i] = self.eta / self.N * (
   #             np.sum(1 - np.cos(np.roll(theta, 1) - theta) + 1 - np.cos(np.roll(theta, -1) - theta)))
   #         self.v[i] += np.mean(self.omegas)
   #         # wy
   #         self.v[i] += theta[i]
   #     return


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