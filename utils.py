import persim
import numpy as np 
import pandas as pd
from math import pi

import os, sys  
import scipy.io as sio
import networkx as nx

from ripser import ripser
from persim.visuals import plot_diagrams

from hausdorff import hausdorff_distance

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [5, 5]

########################################################
############## functions for loading data ##############
########################################################

def load_data(subj_directory, fixed_subj_prefix, task):

    FC_all_subjs = np.load(subj_directory + fixed_subj_prefix + task + '.npy')
    FC_all_subjs = FC_all_subjs * (FC_all_subjs>0) # remove all negative numbers and set to 0
    GA_FC_task = np.mean(FC_all_subjs,0)
    np.save(f'./graphs/global_graph_{task}.npy', GA_FC_task)

    return GA_FC_task

def load_individual_data(subj_directory, fixed_subj_prefix, task):

    FC_all_subjs = np.load(subj_directory + fixed_subj_prefix + task + '.npy')
    FC_all_subjs = FC_all_subjs * (FC_all_subjs>0) # remove all positive numbers and set to 0
    np.save(f'./graphs/individual_graph_{task}.npy', FC_all_subjs)

    return FC_all_subjs

def consolidate_graph(subj_directory, fixed_subj_prefix, task, parcellation):
    GA_FC_task = load_data(subj_directory, fixed_subj_prefix, task)
    n = len(parcellation) # number of regions
    k = np.max(parcellation) + 1 # max number in the yeoROIs - number of ROIs
    Q = np.zeros((k,k)) #consolidated graph for a given fMRI task

    for i in range(n):
        for j in range(n):
            Q[parcellation[i],parcellation[j]] = Q[parcellation[i],parcellation[j]] + GA_FC_task[i,j] 
    row_sums = Q.sum(axis=1)
    # Q_normalized = Q / row_sums[:, np.newaxis]
    Q_normalized = Q / np.max(Q)
    np.fill_diagonal(Q_normalized, 1, wrap=False)
    np.save(f'./graphs/consolidated_graph_{task}.npy', Q_normalized)

    return Q_normalized

#########################################################################
############## functions for computing persistent homology ##############
#########################################################################

def persistent_homology_global(fMRI_tasks, subj_directory, fixed_subj_prefix):

    dgms_list = []

    for i, task in enumerate(fMRI_tasks):
        print(task)
        GA_FC_task = load_data(subj_directory, fixed_subj_prefix, task)
        # print(task, np.min(GA_FC_task), np.max(GA_FC_task))

        dist = 1.0 - GA_FC_task
        dgms = ripser(dist, coeff=2, maxdim=2, distance_matrix=True)['dgms']

        dgms_list.append(dgms)

        if i==0: plot_diagrams(dgms, title=task, show=True) # plot the first task

        plot_diagrams(dgms, title=task, show = False)
        plt.savefig(os.path.join('./figures/', 'ga_global_' + task + '.pdf'))
        plt.close()

        np.save(os.path.join('./results/', 'ga_global_' + task + '.npy'), np.array(dgms, dtype=object), allow_pickle=True)

    return dgms_list

def persistent_homology_individual(fMRI_tasks, subj_directory, fixed_subj_prefix, individual):

    dgms_list = []

    for i, task in enumerate(fMRI_tasks):
        print(task)
        GA_FC_task = load_individual_data(subj_directory, fixed_subj_prefix, task)
        GA_FC_task = GA_FC_task[individual, :, :]
        # print(task, np.min(GA_FC_task), np.max(GA_FC_task))

        dist = 1.0 - GA_FC_task
        dgms = ripser(dist, coeff=2, maxdim=2, distance_matrix=True)['dgms']

        dgms_list.append(dgms)

        if i==0: plot_diagrams(dgms, title=task, show=True) # plot the first task

        plot_diagrams(dgms, title=task, show = False)
        plt.savefig(os.path.join('./figures/', 'ga_global_' + task + '.pdf'))
        plt.close()

        np.save(os.path.join('./results/', 'ga_global_' + task + '.npy'), np.array(dgms, dtype=object), allow_pickle=True)

    return dgms_list

def persistent_homology_consolidated(fMRI_tasks, parcellation, subj_directory, fixed_subj_prefix):

    dgms_list = []

    for i, task in enumerate(fMRI_tasks):
        print(task)
        Q_normalized = consolidate_graph(subj_directory, fixed_subj_prefix, task, parcellation)
        # print(task, np.min(Q_normalized), np.max(Q_normalized))

        dist = 1.0 - Q_normalized
        dgms = ripser(dist, coeff=2, maxdim=2, distance_matrix=True)['dgms']

        dgms_list.append(dgms)

        if i==0: plot_diagrams(dgms, title=task, show=True) # plot the first task

        plot_diagrams(dgms, title=task, show = False)
        plt.savefig(os.path.join('./figures/', 'ga_consolidated_' + task + '.pdf'))
        plt.close()

        np.save(os.path.join('./results/', 'ga_consolidated_' + task + '.npy'), np.array(dgms, dtype=object), allow_pickle=True)

    return dgms_list

def persistent_homology_subnet(fMRI_tasks, parcellation, subj_directory, fixed_subj_prefix, FNs):

    dgms_subnet_all = []

    for i, task in enumerate(fMRI_tasks):
        print(task)
        GA_FC_task = load_data(subj_directory, fixed_subj_prefix, task)
        # print(task, np.min(GA_FC_task), np.max(GA_FC_task))

        parc_order = np.argsort(parcellation)
        GA_FC_order = GA_FC_task[parc_order,:]
        GA_FC_order = GA_FC_order[:,parc_order]

        dgms_list = []

        for j in range(8):
            FC_comm = GA_FC_task[parcellation==j,:]
            FC_comm = FC_comm[:,parcellation==j]

            dist = 1.0 - FC_comm
            dgms = ripser(dist, coeff=2, maxdim=2, distance_matrix=True)['dgms']

            dgms_list.append(dgms)

        if i==0: plot_homology_subnet(dgms_list, FNs, task, plot = True) # plot the first task

        plot_homology_subnet(dgms_list, FNs, task, plot = False)

        np.save(os.path.join('./results/', 'ga_subnet' + task + '.npy'), np.array(dgms_list, dtype=object), allow_pickle=True)

        dgms_subnet_all.append(dgms_list)

    return dgms_subnet_all

##############################################################
############## functions for computing distance ##############
##############################################################
def compute_hausdorff(dgms, n, save_name):
    
    homology_order = 0
    
    #Zeroth homology: Compute distance between different tasks
    d0 = np.zeros((n,n))
    for i2 in range(n): #8 func networks
        for i3 in range(n): #same FN dist=0
            d0[i2,i3] = hausdorff_distance(dgms[i2][homology_order], dgms[i3][homology_order])

    np.save(os.path.join('./results/', f'Hausdorff_H0_{save_name}.npy'), d0)

    return d0

def compute_wasserstein(dgms, homology_order, n, save_name):

    dis = np.zeros((n,n))
    for i2 in range(n): #8 func networks
        for i3 in range(n): #same FN dist=0
            dis[i2,i3] = persim.sliced_wasserstein(dgms[i2][homology_order],dgms[i3][homology_order],300);

    np.save(os.path.join('./results/', f'Wasserstein_H{homology_order}_{save_name}.npy'), dis)

    return dis


################################################
############## functions for plot ##############
################################################

########## plots for explore cocycle ##########
def drawLineColored(X, C):
    for i in range(X.shape[0]-1):
        # plt.plot(X[i:i+2, 0], X[i:i+2, 1], c=C[i, :], lineWidth = 3)
        plt.plot(X[i:i+2, 0], X[i:i+2, 1], c=C[i, :])

def plotCocycle2D(D, X, cocycle, thresh):
    """
    Given a 2D point cloud X, display a cocycle projected
    onto edges under a given threshold "thresh"
    """
    #Plot all edges under the threshold
    N = X.shape[0]
    t = np.linspace(0, 1, 10)
    c = plt.get_cmap('Greys')
    C = c(np.array(np.round(np.linspace(0, 255, len(t))), dtype=np.int32))
    C = C[:, 0:3]

    for i in range(N):
        for j in range(N):
            if D[i, j] <= thresh:
                Y = np.zeros((len(t), 2))
                Y[:, 0] = X[i, 0] + t*(X[j, 0] - X[i, 0])
                Y[:, 1] = X[i, 1] + t*(X[j, 1] - X[i, 1])
                drawLineColored(Y, C)
    #Plot cocycle projected to edges under the chosen threshold
    for k in range(cocycle.shape[0]):
        [i, j, val] = cocycle[k, :]
        if D[i, j] <= thresh:
            [i, j] = [min(i, j), max(i, j)]
            a = 0.5*(X[i, :] + X[j, :])
            plt.text(a[0], a[1], '%g'%val, color='b')
    #Plot vertex labels
    for i in range(N):
        plt.text(X[i, 0], X[i, 1], '%i'%i, color='r')
    plt.axis('equal')

########## plot homology for functional networks ##########
def plot_homology_subnet(ga_subnet, FNs, task, plot = False):
    plt.figure(figsize=(15,10))

    for i in range(8):
        plt.subplot(2,4,i+1)
        plot_diagrams(ga_subnet[i])
        plt.ylim(0,1)
        plt.xlim(-0.1,1)
        plt.title(FNs[i], fontsize=10)
    plt.tight_layout()    
    plt.savefig(os.path.join('./figures/', 'ga_subnet_' + task + '.pdf'),dpi=300) 
    if plot: 
        print("plot")
        plt.show()
    plt.close()

########## plot distance in 2d ##########
def plot_hausdorff_distance(d0, bin_labels, save_name):
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    x_pos = np.arange(len(bin_labels))

    shw0 = ax.imshow(d0, cmap='hot')
    plt.sca(ax)
    plt.xticks(x_pos,bin_labels, fontsize = 35,rotation = 45)
    plt.yticks(x_pos,bin_labels, fontsize = 35,rotation = 45)
    # ax[0].title.set_text('H_1 Wasserstein Distance of FNs')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    # make bar
    bar = plt.colorbar(shw0)
    bar.set_ticks([])
    # bar.set_label('Hausdorff Distance', fontsize=25)
    # bar.ax.tick_params(labelsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('./figures/', f'H0_distance_{save_name}.pdf'),dpi=300) 
    

def plot_wasserstein_distance(dis, homology_order, bin_labels, save_name):
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    x_pos = np.arange(len(bin_labels))

    # ax2 = fig.add_subplot(112)
    shw1 = ax.imshow(dis, cmap='hot')
    plt.sca(ax)
    plt.xticks(x_pos,bin_labels, fontsize = 35,rotation = 45)
    plt.yticks(x_pos,bin_labels, fontsize = 35,rotation = 45)
    # ax[0].title.set_text('H_1 Wasserstein Distance of FNs')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    # make bar
    bar = plt.colorbar(shw1)
    bar.set_ticks([])
    # bar.set_label('Hausdorff Distance', fontsize=25)
    # bar.ax.tick_params(labelsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('./figures/', f'H{homology_order}_distance_{save_name}.pdf'),dpi=300) 


########## plot distance in 1d (averaged) ##########
def stylize_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(right='off', direction='out', width=1)

def plot_averaged_distance(dis, bin_labels, save_name):
    dis_avg = np.average(dis, axis=0)/2

    fig, ax = plt.subplots(1,1,figsize=(5,10))
    # plot bar plot
    x_pos = np.arange(len(bin_labels))
    ax.bar(x_pos, dis_avg, color=(0.1, 0.1, 0.1, 0.1),  edgecolor='blue')
    stylize_axes(ax)
    plt.sca(ax)
    plt.xticks(x_pos,bin_labels, fontsize = 35,rotation = 90)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.yticks(fontsize = 35)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    fig.tight_layout()
    fig.savefig(os.path.join('./figures/', f'avg_dis_{save_name}.pdf'),dpi=300) 

def plot_variance(dis, bin_labels, save_name):
    dis_var = np.var(dis,axis=0)

    fig, ax = plt.subplots(1,1,figsize=(10,10))
    df = pd.DataFrame(dis_var)
    # number of variable
    categories= bin_labels
    N = len(categories)
    
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values=df.iloc[:,0].values.flatten().tolist()
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='black', size=20)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    # plt.yticks([np.min(values),np.average(values),np.max(values)], color="grey", size=10)
    plt.ylim(0,np.max(values))
    plt.xticks(fontsize = 25)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.yticks(fontsize = 7)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)

    # Show the graph
    plt.show()
    fig.tight_layout()
    fig.savefig(os.path.join('./figures/', f'var_dis_{save_name}.pdf'),dpi=300) 