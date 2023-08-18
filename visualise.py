import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as am
from mpl_toolkits.axes_grid1 import make_axes_locatable

import agent



def plot1D_frame(model,t,save=True):
    fig,axs = plt.subplots(1,2)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    color = 'tab:red'

    phases = model.pick(t)
    agent.shift(phases)

    axs[0].set_ylabel('$\Theta$', color=color)
    axs[0].plot(phases, color=color)
    axs[0].tick_params(axis='y', labelcolor=color)
    #ax2 = axs[0].twinx()  # instantiate a second axes that shares the same x-axis
    #color = 'tab:blue'
    #ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    #ax2.plot(model.omegas, color=color,linestyle='None',marker='x')
    #ax2.tick_params(axis='y', labelcolor=color)

    axs[0].plot(phases)
    gradient = (phases[len(phases)-1]-phases[0])/len(phases)
    k = gradient/(2*np.pi/len(phases))
    print(f'{k=}')
    axs[0].set_xlabel('$x$')
    axs[0].set_title(r'spatial phase distribution $t={}$, $\sigma={}, \eta ={}, bc ={}$'.format(model.tmax-t,model.sigma, model.eta,model.grad))

    im1 = axs[1].imshow(model.theta.T %(2*np.pi), cmap='twilight',aspect='auto',interpolation='none')
    # plt.colorbar(im1,orientation='vertical')
    portion = model.theta[int(len(phases)/2)][int(9*model.tmax/10):model.tmax]
    time_d = -1*(portion[0] - portion[-1]) / len(portion)
    print(f'{time_d=}')
    if k!=0 :
        c = time_d/gradient
        print(f'{c=}')
        print(c/(model.eta)-gradient)
        print(c**2/model.eta)

        if model.eta != 0:
            fac = (len(phases)*c)/(model.eta*2*np.pi)
            print(f'{fac=}')
    cbar = fig.colorbar(im1, ticks=[0, np.pi, 2*np.pi], orientation='vertical')
    cbar.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    axs[1].set_xlabel('$x$')
    axs[1].set_ylabel('$t$')
    axs[1].set_title('space time graph of phase evolution $\sigma={}, \eta ={}$'.format(model.sigma, model.eta))
    title = get_title(model,variable='phase',count=model.count) + r'_t={}'.format(t)
    if save:
        plt.savefig('frames/' +title + '.png')
    plt.show()

def plot2D_frame(t,model,save=True):
    if model.dim =='temp_noise':
        res = 100
    else:
        res=1
    print(np.shape(model.theta))
    im2 = plt.imshow(model.theta.T[int(t/res)-1].reshape((model.N,model.M))% (2*np.pi), cmap='twilight',vmin=0, vmax=2*np.pi,interpolation='none')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(r'$t={}$, $\sigma={}, \eta ={}, bc = {}$'.format(t,model.sigma, model.eta,model.grad))
    #plt.colorbar(im2,orientation= "horizontal")
    cbar = plt.colorbar(im2, ticks=[0, np.pi, 2 * np.pi], orientation='horizontal')
    cbar.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

    #title = r'2D_tse_{}_{}_{}'.format(t,model.sigma, model.eta)
    title = get_title(model, variable='phase',count=model.count) + r'_t={}'.format(int(t/res)-1)
    if save:
        plt.savefig('frames/' + title + '.png')
    plt.show()


def plot_branch(t,model,save=True):
    theta, branch = model.theta.T[t].reshape(2, model.N * model.M)
    fig,axs = plt.subplots(2)
    im1 = axs[0].imshow(theta.reshape((model.N,model.M))% (2*np.pi), cmap='twilight',vmin=0, vmax=2*np.pi,interpolation='none')
    im2 = axs[1].imshow(branch.reshape((model.N, model.M)) % (2 * np.pi), cmap='twilight', vmin=0, vmax=2 * np.pi,
                  interpolation='none')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(r'$t={}$, $\sigma={}, \eta ={}, bc = {}$'.format(t,model.sigma, model.eta,model.grad))
    #plt.colorbar(im2,orientation= "horizontal")
    cbar = plt.colorbar(im1, ticks=[0, np.pi, 2 * np.pi], orientation='horizontal')
    cbar.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    plt.show()

def plot_avg(repeats,t):
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    phases = repeats.avgtheta[t]
    #agent.shift(phases)
    axs[0].plot(phases)
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$\Theta$')
    axs[0].set_title(r'spatial phase distribution $t={}$, $\sigma={}, \eta ={}, bc ={}$'.format(t, repeats.sigma, repeats.eta,
                                                                                   repeats.grad))

    im1 = axs[1].imshow(repeats.avgtheta % (2 * np.pi), cmap='twilight', aspect='auto', interpolation='none')
    # plt.colorbar(im1,orientation='vertical')
    cbar = fig.colorbar(im1, ticks=[0, np.pi, 2 * np.pi], orientation='vertical')
    cbar.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    axs[1].set_xlabel('$x$')
    axs[1].set_ylabel('$t$')
    axs[1].set_title('space time graph of phase evolution $\sigma={}, \eta ={}$'.format(repeats.sigma, repeats.eta))
    plt.show()

    return


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
    else:
        print('Error bc not present')
        return
    title = r'2D_{}_{}_{}_{}'.format(val, t, model.sigma, model.eta)
    if save:
        plt.savefig(title + '.png')
    plt.show()


def animate(model,rem,variable='phase'):
    res=1
    fig = plt.figure()
    #print(model.theta.T[].reshape((20,128))% (2*np.pi))
    if variable == 'phase':
        a0 = model.theta.T[0].reshape((model.N, model.M)) % (2 * np.pi)
        # a = model.theta.T[t].reshape((model.N, model.M)) % (2 * np.pi)
    elif variable == 'branch':
        a0,b0 = model.theta.T[0].reshape(2,model.N, model.M) %(2*np.pi)
    elif variable == 'vorticity':
        model.find_vorticity_2()
        a0 = model.vorticity[0]
        a = model.vorticity# % (2 * np.pi)
    elif variable == 'energy':
        model.find_energy()
        a0 = model.energy[0]
        a = model.energy
    elif variable == 'divergence':
        model.find_divergence()
        a0 = model.divergence[0]
        a = model.divergence
    elif variable == 'vort_divergence':
        model.find_vorticity_2(div=True)
        a0 = model.vorticity[0]
        a = model.divergence
    elif variable == 'v_x':
        a0 = model.v_x[0]
        a = model.v_x
    elif variable == 'v_y':
        a0 = model.v_y[0]
        a = model.v_y
    elif variable == 'velocity':
        a0 = model.velocity[0]
        a = model.velocity
    else:
        print('Error variable not present')
        a=0

        return
    if variable == 'branch':
        fig, axs = plt.subplots(2)
        im = axs[0].imshow(a0, cmap='twilight', vmin=0, vmax=2*np.pi, interpolation='none')
        im2 = axs[1].imshow(b0, cmap='twilight', vmin=0, vmax=2*np.pi, interpolation='none')
    elif variable == 'vorticity' or variable == 'velocity' or variable=='energy' or variable == 'divergence' or variable == 'vort_divergence':
        im = plt.imshow(a0, cmap='PiYG',  interpolation='none')
    else:
        im = plt.imshow(a0, cmap='twilight', vmin=0, vmax=2*np.pi, interpolation='none')

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.colorbar(im, orientation="horizontal")

    def update(t):
        if variable == 'phase':
            im.set_array(model.theta.T[t].reshape((model.N,model.M)) % (2*np.pi))
        elif variable == 'velocity':
            if t == len(model.velocity):
                im.set_array(model.velocity[t-1])
            else:
                im.set_array(model.velocity[t])
        elif variable == 'branch':
            a, b = model.theta.T[t].reshape(2,model.N, model.M) % (2*np.pi)
            im.set_array(a)
            im2.set_array(b)
        elif variable == 'vorticity' or variable == 'vort_divergence':
            a = model.vorticity #% (2 * np.pi)
            im.set_array(a[t])
        elif variable == 'energy':
            a = model.energy
            im.set_array(a[t])
        elif variable == 'divergence':
            a = model.divergence
            im.set_array(a[t])
        plt.title(r'${}, t={}$, $\sigma={}, \eta ={}$'.format(variable,t, model.sigma, model.eta))
        return [im]
    if model.dim == 'temp_noise':
        res = 100
        print('temp_noise anim')
        anim = am.FuncAnimation(fig, update, frames=[i for i in range(0,int(model.tmax/res)-1)], repeat=False)
    else:
        anim = am.FuncAnimation(fig, update, frames=[i for i in range(0,model.tmax)], repeat=False)


    #title = r'2Danim_{}_{}_{}_{}'.format(variable, val, model.sigma, model.eta)
    title = get_title(model, variable,model.count)
    if rem:
        title = title + '_alt'
    factor = 1/(model.Dt)#*res)
    anim.save('animations/' + title + '.mp4', fps=25*factor, extra_args=['-vcodec', 'libx264'])
    print(str(variable) + ' animation saved')
    return title

def get_title(model,variable='phase',count=0):
    if model.bc == 'grad':
        if model.grad == [1, 1]:
            val = 'A'
        elif model.grad == [-1, 1]:
            val = 'B'
        elif model.grad == [1, -1]:
            val = 'C'
        elif model.grad == [0, 0]:
            val = 'D'
        else:
            val = str(model.grad)
    elif model.bc == 'fix':
        val = 'fix'
    elif model.bc == 'periodic':
        val = 'p'
    elif model.bc == 'flat':
        val = 'flat'
    else:
        print('Error bc not present')
        return
    if model.dim == 'temp_noise':
        title = r'temporal/{}_{}_{}_{}_{}_{}_{}'.format(variable+str(model.tmax), model.dim + str(model.N), val, model.sigma, model.eta,model.W,str(model.grad))
    else:
        title = r'{}_{}_{}_{}_{}_{}'.format(variable+ str(model.tmax), model.dim+ str(model.N), val, model.sigma, model.eta,str(model.grad))
    if count > 0:
        title = title + '_v' + str(count)
    return title
