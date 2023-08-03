import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import visualise

factor = 0.1
E = 1
tmax = 500
L=128

psi_init = np.random.random(128)
field = np.zeros([2,L])
field.T[0] = [0,0.1]
def lifschitz(t,psi):
    print(t)
    laplacian = np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) - 2*psi
    prossimo = np.zeros(len(psi))
    prossimo[:] = -0.5*laplacian[:]*t - factor*psi[:]**3*t

    return psi



def plot1D_frame(psi,t,save=True):
    fig,axs = plt.subplots()
    color = 'tab:red'
    phases = psi[t]
    axs.set_ylabel('$\Psi$', color=color)
    axs.plot(phases, color=color)
    axs.tick_params(axis='y', labelcolor=color)
    #ax2 = axs[0].twinx()  # instantiate a second axes that shares the same x-axis
    #color = 'tab:blue'
    #ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    #ax2.plot(model.omegas, color=color,linestyle='None',marker='x')
    #ax2.tick_params(axis='y', labelcolor=color)

    axs.plot(phases)
    gradient = (phases[len(phases)-1]-phases[0])/len(phases)
    k = gradient/(2*np.pi/len(phases))
    print(f'{k=}')
    axs.set_xlabel('$x$')
    axs.set_title(r'spatial phase distribution, ')

    #im1 = axs[1].imshow(psi.T %(2*np.pi), cmap='twilight',aspect='auto',interpolation='none')
    # plt.colorbar(im1,orientation='vertical')


    #cbar = fig.colorbar(im1,  orientation='vertical')
    #axs[1].set_xlabel('$x$')
    #axs[1].set_ylabel('$t$')
    #axs[1].set_title('space time graph of psi evolution ')
    plt.show()
    return

def solve(dx):
    psi = field[0]
    u = field[1]
    print('hi')
    for x in range(len(psi) - 1):
        print(psi[x])
        u[x + 1] = 2 * (-factor * psi[x] ** 3 - E * psi[x]) * dx + u[x]
        psi[x + 1] = psi[x] + u[x] * dx

    print(psi)
sol = solve_ivp(lifschitz, [0, tmax], psi_init, method='LSODA', t_eval=range(tmax)).y.T
np.shape(sol)
plt.plot(sol[tmax-1])
plt.show()