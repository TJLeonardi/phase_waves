import numpy as np
import analyse
import matplotlib.pyplot as plt

def all_sites(model):
    all_sites = np.array([])
    for t in range(len(model.theta)):
        np.append(all_sites,analyse.vorsites(model,t))
    return all_sites

def pair_distances(model):
    model.find_vorticity_2()
    vortex = np.argwhere(model.vorticity >0.5)
    antivortex = np.argwhere(model.vorticity <-0.5)
    dists = np.zeros(len(vortex))
    for frame in range(len(vortex)):
        if vortex[frame][1]-antivortex[frame][1] >= model.N:
            x = (vortex[frame][1]-antivortex[frame][1])
        else:
            x = vortex[frame][1]+antivortex[frame][1] - model.N
        if vortex[frame][2]-antivortex[frame][2] >= model.M:
            y = (vortex[frame][2]-antivortex[frame][2])
        else:
            y = vortex[frame][2]+antivortex[frame][2] - model.M
        dists[frame] = np.sqrt(x**2 + y**2)
    return dists

def next_frame(model,oldstate,frame1,frame2=0,lim=1):
    frame2 = frame1 + 1
    #sites1 = analyse.vorsites(model,frame1)
    sites1 = oldstate
    sites2 = analyse.vorsites(model,frame2)
    results = np.array([])
    born = np.array([])
    sites2copy = sites2
    for i in range(len(sites1)):
        nextcells = np.array([])
        vort_i = sites1[i][3]
        if np.isnan(sites1[i].any()):
            np.vstack([results,[frame2,np.nan,np.nan,np.nan]])
            continue
        shortest = 64
        nearest_j=-1
        distances= np.array([])
        for j in range(len(sites2)):
            vort_j = sites2[j][3]
            j_distance = np.sqrt( ((sites2[j][1] - sites1[i][1]) ) ** 2 + ((sites2[j][2] - sites1[i][2]) ) ** 2)
            if np.sign(vort_i) == np.sign(vort_j):
                if j_distance <= lim and  np.sign(vort_i) == np.sign(vort_j):
                    #if np.abs((sites2[j][1]-sites1[i][1])%model.N) <= lim and np.abs((sites2[j][2]-sites1[i][2])%model.M) <= lim and vort_i == vort_j:
                    distances = np.append(distances, [j_distance,j])
                    if j_distance <= shortest:
                        shortest=j_distance
                        nearest_j = j
                    #print(r'distances {}'.format(len(distances)))
                    if len(nextcells) == 0:
                        nextcells = [[frame2,sites2[j][1],sites2[j][2],vort_i]]

                    else:
                        nextcells = np.vstack([nextcells,[frame2,sites2[j][1],sites2[j][2],vort_i]])
        #if len(distances) > 1:
            #print(f'{distances=}')


        #nextcells = [[frame2,sites2[nearest_j][1],sites2[nearest_j][2],vort_i]]
        if len(nextcells) == 1:
            sites2copy.remove(nextcells[0])
            if len(results) == 0:
                results = nextcells[0]
            else:
                results = np.vstack([results,nextcells[0]])
        elif len(nextcells) == 0:
            if len(results) == 0:
                results = np.array([frame2,np.nan,np.nan,np.nan])
            else:
                results = np.vstack([results,[frame2,np.nan,np.nan,np.nan]])
        else:
            #nearest_j = np.argmin(distances)
            #print(f'{i=}')
            #print(f'{nearest_j=}')
            #print(nextcells[nearest_j])
            #print(f'{sites2copy=}')
            #print(f'{nextcells=}')
            #print(f'{sites2[nearest_j]=}')
            nearest_nextcells = [frame2, sites2[nearest_j][1], sites2[nearest_j][2], vort_i]
           #results = np.vstack([results,nextcells[nearest_j]])
            if len(results) == 0:
                results = np.array(nearest_nextcells)
            else:
                results = np.vstack([results,nearest_nextcells])
            #print(f'{nearest_nextcells=}')

            #sites2copy.remove(list(nextcells[nearest_j]))
            sites2copy.remove(nearest_nextcells)
            #print(f'{nextcells=}')
            #for k in range(1,len(nextcells)):
            #    if len(born) == 0:
            #        print('gone')
                    #born = #nextcells[k]
            #    else:
             #       print('gone')
                    #born = np.vstack([born,nextcells[k]])
                    #sites2copy.remove(list(nextcells[k]))
    #print(sites2copy)
    #print(r'distances {}'.format(len(results)))
    if len(sites2copy) > 0:
        #born = sites2copy
        print(r'sites2copy {}'.format(len(sites2copy)))
        if len(born) == 0:
            born = sites2copy
        else:
            for k in range(len(sites2copy)):
                born = np.vstack([born, sites2copy[k]])
    if len(born) >0:
        print(r'born {}'.format(len(born)))
        newstate = np.concatenate([results,born])
    else:
        newstate= results

    return newstate,frame2


def states(model,time,t0=0,lim=1):
    states = np.array([analyse.vorsites(model,t0)])

    for t in range(t0,time):
        output = next_frame(model,states[t-t0],t,0,lim)
        born = len(output[0]) - len(states[t-t0])
        if born != 0:
            preborn = np.zeros([born,t-t0+1,4])
            for t_mid in range(t-t0+1):
                #if t<time-1:
                preborn.transpose((1,0,2))[t_mid][:] = np.array([t_mid+t0,np.nan,np.nan,np.nan])
            #print(preborn.shape)
            #print(states.transpose((1,0,2)).shape)
            states = np.concatenate([states.transpose((1,0,2)),preborn])
            #with np.printoptions(threshold=np.inf):
            #    print(states)
        else:
            states=states.transpose((1,0,2))

        states = np.concatenate([states.transpose((1,0,2)),[output[0]]])#np.append(states,[output[0]]).T.reshape(t-t0+2,len(output[0]),4)
    return states

def plot_trajectories(model,vortices):

    #with np.printoptions(threshold=np.inf):
    #    print(vortices)
    for i in range(len(vortices)):
        #print(len(vortices[i][:,1]))
        if vortices[i][0, 3] > 0:
            marker = '+'
        else:
            marker = 'x'
        plt.plot(vortices[i][:,2],vortices[i][:,1],marker=marker)
    ax = plt.gca()

    plt.xlim(0, model.N-1)
    plt.ylim(0, model.M-1)
    plt.gca().invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    plt.show()