import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dist_measure = {
    'Euc': lambda x1, x2: np.linalg.norm(x1-x2),
    'cos': lambda x1, x2: np.linalg.norm(x1/np.linalg.norm(x1) - x2/np.linalg.norm(x2))**2,
}

def reorder(dist_matrix):
    '''
    Reorder the groups so that similar groups are close in the list.
    
    :param
    :return
    '''
    dist = np.copy(dist_matrix)
    n = len(dist)
    mask = np.zeros((n,))
    mask_val = np.max(dist)+1.0

    # ignore the distences on diagonal
    for i in range(n):
        dist[i,i] = np.Inf
        
    draw = np.argmin(np.min(dist, axis=0))
    indices = [draw]
    mask[draw] = mask_val
    for i in range(n-1):
        draw = np.argmin(dist_matrix[draw]+mask)
        indices.append(draw)
        mask[draw] = mask_val
    return indices

def cluster(point_list, n_cluster, dist='Euc'):
    '''
    Use k-means to find cells with similar trend.
    The returned lists are ordered so that similar groups are close.
    
    :param ndarray(float) point_list: Shape = (n_point, n_feature)]
    :param int n_cluster: For k-means
    :return tuple(list, list): reordered groups and centers:
        groups: list(list(int))
        centers: list(ndarray(shape=(n_feature,)))
    '''
    # k-means
    kmeans = KMeans(n_cluster, random_state=0, n_jobs=-1).fit(point_list)
    dist_s2c = kmeans.inertia_/len(point_list)
    print('average of squared distances of samples to centers:', dist_s2c)
    centers = kmeans.cluster_centers_
    m_pairs_squared = ((centers[None, :, :] - centers[:, None, :])**2)
    dist_c2c = m_pairs_squared.sum() / (n_cluster*(n_cluster-1))
    print('average of squared distances of centers to centers:', dist_c2c)
    print('ratio:', dist_c2c/dist_s2c)
    
    # calculate distances
    dist = dist_measure[dist]
    dist_matrix = np.zeros((n_cluster, n_cluster))
    for i in range(n_cluster):
        for j in range(n_cluster):
            dist_matrix[i,j] = dist(centers[i], centers[j])
    
    # reorder the groups
    group_idx = reorder(dist_matrix)
    
    # collect groups
    groups = []
    for i in range(n_cluster):
        groups.append([])
    for i, l in enumerate(kmeans.labels_):
        groups[l].append(i)
        
    # reorder the cells in each group
    for n in range(n_cluster):
        g = groups[n]
        n_local = len(g)
        dist_local = np.zeros((n_local, n_local))
        for i in range(n_local):
            for j in range(n_local):
                dist_local[i,j] = dist(point_list[g[i]], point_list[g[j]])
        indices = reorder(dist_local)
        groups[n] = [g[i] for i in indices]
    
    reordered_groups = []
    reordered_centers = []
    for i in group_idx:
        reordered_groups.append(groups[i])
        reordered_centers.append(centers[i])
    return reordered_groups, reordered_centers

def plot_weights(W, groups=None, perm_dim=(True, True), figsize=(32,28)):
    # concatenate groups and reorder cells
    if groups:
        indices = []
        seps = [] # separations
        ticks, labels = [], []
        accu = 0
        for g, group in enumerate(groups):
            indices += group
            accu += len(group)
            seps.append(accu)
            ticks.append(accu-len(group)/2)
            labels.append(f'{g}: {len(group)}')
        seps = seps[:-1]
        if perm_dim[0]:
            W = W[indices,:]
        if perm_dim[1]:
            W = W[:,indices]
        
    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    vmax = np.max(np.abs(W))
    plt.imshow(W, cmap='bwr', vmax=vmax, vmin=-vmax)
    plt.colorbar().ax.tick_params(labelsize=48)
    
    # add separation lines
    if groups:
        for sep in seps:
            if perm_dim[0]:
                plt.axhline(y=sep-0.5, color='black', linewidth=1.0)
            if perm_dim[1]:
                plt.axvline(x=sep-0.5, color='black', linewidth=1.0)
        plt.xticks(fontsize=48)
        plt.yticks(fontsize=48)
        if perm_dim[0]:
            plt.yticks(ticks, labels)
            plt.ylabel('Group ID: Group size', fontsize=48)
    
    plt.savefig('weight_.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.show()

def analyze_weights(W, n_cluster=15, dist='cos'):
    groups, centers = cluster(W.transpose(), n_cluster, dist) # use dim0 of W as features
    plot_weights(W, groups)
    return groups

def plot(cells, picked='all', show_legend = False, ylim=(0, 1)):
    '''
    Plot line chart of cells in the set "picked".
    
    :param ndarray(float) cells: Shape = (128, s)
    :param set(int) picked: Indices of "cells"
    :param bool show_legend: For plotting
    :param tuple(float) ylim: For plotting
    '''
    if picked=='all':
        picked = set(range(cells.shape[0]))
    print('cells:', sorted(list(picked)))
    
    x = np.arange(cells.shape[1])
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.xlim((0, cells.shape[1]))
    plt.ylim(ylim)
    for i, cell in enumerate(cells):
        if i in picked:
            plt.plot(x, cell, label=f'cell {i}')
    if show_legend:
        leg = ax.legend(loc='upper right', shadow=True)
    plt.title(f'{len(picked)} cells')
    plt.show()
    
def hist(diff, max_diff):
    '''
    Plot histogram of "diff".
    
    :param array(float) diff: Shape = (n_cell, )
    :param float max_diff: For plotting
    '''
    fig, ax = plt.subplots(figsize=(16, 3))
    n, bins, patches = plt.hist(diff, bins=20, range=(0, max_diff))
    ax.set_xticks(bins)
    for i in range(len(n)):
        plt.text(bins[i], n[i], str(n[i]))

def pick(cells, feature, mask=None, criterion=lambda d: True, max_diff=0.5, include='all'):
    '''
    Get the set of cells close to "feature" and satisfying "criterion" weighted by "mask".
    
    :param ndarray(float) cells: Shape = (128, s)
    :param array(float) feature: Shape = (s, )
    :param array(float) mask: Shape = (s, )
    :param lambda float->bool criterion: Reture True if diff is in the desired range
    :return set(int): picked indices
    '''
    feature = np.array(feature)
    mask = np.ones(cells.shape[1]) if mask==None else np.array(mask)
    
    if include=='all':
        include = set(range(cells.shape[0]))
    
    diffs = np.zeros(cells.shape[0])
    picked = set()
    for i, cell in enumerate(cells):
        diffs[i] = np.sqrt(np.sum(np.square(cell-feature)*mask)/np.sum(mask))
        if (i in include) and criterion(diffs[i]):
            picked.add(i)
    others = set(range(len(cells))) - picked
    
    hist([diffs[i] for i in picked], max_diff)
    plt.title(f'picked: {len(picked)}')
    hist([diffs[i] for i in others], max_diff)
    plt.title(f'others: {len(others)}')
    plot(cells, picked=picked, show_legend=True)
    return picked

def analyze(cells, n_cluster, ylim=(0, 1)):
    '''
    Plot all groups in "cells".
    
    :param ndarray(float) cells: Shape = (128, s)
    :param int n_cluster: For k-means
    :param tuple(float) ylim: For plotting
    '''
    groups, centers = cluster(cells, n_cluster)
    for i, group in enumerate(groups):
        print('-----------------------------')
        print(f'group {i}: {centers[i]}')
        plot(cells, picked=group, show_legend=True, ylim=ylim)

def draw(string):
    '''
    Draw a character from "string".
    
    :param str string: string to be draw from
    '''
    import random
    idx = random.randrange(len(string))
    drawed = string[idx]
    print(drawed)
    return drawed