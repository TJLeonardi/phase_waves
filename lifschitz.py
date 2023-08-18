import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import visualise
from scipy.integrate import solve_bvp

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
sigma = 0.1
def function(x,y,p):
    K = p[0]
    E_0 = - 0.5*K**2
    E = E_0 - sigma / (2 * K) ## ground state solution
    E_0 = -1      ### trial periodic solution
    E = p[0]

    factor = sigma/(E_0-E)
    out = np.vstack((y[1], -2*E_0*y[0]-2*factor*y[0]**3))
    return out
def bc(ya,yb,p):
    K = p[0]
    #grad = (0.5*K)**0.5 * K * np.tanh(K)/np.cosh(K)
    #grad = 0.3
    grad= 1
    return np.array([ya[0]-yb[0],ya[1]-grad,yb[1]-grad])

x = np.linspace(-1, 1, 5)
y_a = np.zeros((2, x.size))

y_b = np.zeros((2, x.size))

y_b[0] = 0.5
p_0 = 1
res_a = solve_bvp(function, bc, x, y_a,p=[p_0])
res_b = solve_bvp(function, bc, x, y_b,p=[p_0])
x_plot = np.linspace(-1, 1, 200)
y_plot_a = res_a.sol(x_plot)[0]
y_plot_b = res_b.sol(x_plot)[0]
fac = res_b.p[0]
print(fac)
y_plot_c = ((fac/2)**0.5) * np.divide(1,np.cosh(fac*x_plot))
y_plot_d = np.log(abs(y_plot_b))
plt.plot(x_plot, y_plot_a, label='y_a')
plt.plot(x_plot, y_plot_b, label='y_b')
plt.plot(x_plot, y_plot_c, label='y_c')
plt.plot(x_plot, y_plot_d, label='y_d (=ln(|y_b|)')


plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()


#sol = solve_ivp(lifschitz, [0, tmax], psi_init, method='LSODA', t_eval=range(tmax)).y.T
#np.shape(sol)
#plt.plot(sol[tmax-1])
#plt.show()

