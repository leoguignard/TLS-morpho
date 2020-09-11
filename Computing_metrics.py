# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (leo.guignard...@AT@...gmail.com)

from scipy.interpolate import InterpolatedUnivariateSpline
from skimage import morphology, measure
from multiprocessing import Pool
from scipy import ndimage as nd
import numpy as np

def Dijkstra(params):
    """
    Computes the distance of all nodes to a given node `s` within a
    provided graph. In that particular instance, we know that nodes
    are 4 connected in 2D and 6-connected in 3D so the distance between
    two consecutive nodes is always 1.

    Parameter
    ---------
    params : list
        So the function can be called in parallel, the parameters are
        embeded into a list. The first element of the list is the set
        of nodes `nodes` of the graph, the second element is a dictionary
        where the keys are nodes from `nodes` mapped onto a list of
        neighbouring nodes also from `nodes`. The third and last element
        of the list is the starting node `s` to which the distances are
        computed

    Returns
    -------
     : int
        the node from `nodes` that is the furthest away from `s`
     : int
        the distance of the furthest node to `s`
    prev : dict
        a dictionary that maps a node to its closest previous one
    """
    nodes, neighb, s = params
    dist = dict(zip(nodes, [np.inf,]*len(nodes)))
    prev = dict(zip(nodes, [-1,]*len(nodes)))
    Q = set(nodes)
    dist[s] = 0
    while 0 < len(Q):
        u = min(Q, key=dist.get)
        Q.remove(u)
        for v in set(neighb[u]).intersection(Q):
            alt = dist[u] + 1
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
    return max(dist, key=dist.get), max(dist.values()), prev

def compute_metrics3D(bin_im, dist_trsf_im, vs=1):
    """
    Given an isotropic 3D binary image `bin_im`, a distance transform image
    `dist_trsf_im` of `bin_im` and a voxel size `vs`, computes the
    length, the width, the aspect ratio, the solidity, the sphericity,
    the volume and the surface of the binary image.
    Notes: - `bin_im` has to be isotropic otherwise the distances won't be
           computed correctly
           - `bin_im` must only contain a single connected component
           - `bin_im` and `dist_trsf_im` must share the same shape

    Parameters
    ----------
    bin_im : array_like
        A n*m*l binary array with one single connected component
    dist_trsf_im : array_like
        A distance transformation of the bin_im array (distance in voxels)
    vs : float
        The size of the isotropic voxel in order to get measurements
        in physical units

    Returns
    -------
    length : float
        The length of the masked object in the binary image
    width_spl : scipy.interpolate.InterpolatedUnivariateSpline
        The interpolated width along the masked object
        in the binary image
    AR : float
        The aspect ratio of the masked object in the binary image
        `length/median(width)`
    solidity : float
        The solidity of the masked object in the binary image
        Ratio of pixels in the region to pixels of the convex hull image.
    volume :
        The volume of the masked object in the binary image
        Number of voxels in the region
    surface :
        The surface of the masked object in the binary image
    sphericity :
        The sphericity of the masked object in the binary image
        (pi**(1/3)*(6*`volume`)**(2/3))/(`surface`)
    """
    dist_trsf_im = dist_trsf_im*vs

    # Computes volume, surface, sphericity, solidity
    props = measure.regionprops(bin_im.astype(np.uint8))[0]
    volume = np.sum(bin_im)*vs**3
    surface = props.area*vs**2
    sphericity = (np.pi**(1/3)*(6*volume)**(2/3))/(surface)
    solidity = props.solidity

    # Skeletonize the mask and retrieve the coordinates of the skeleton
    skel_im = morphology.skeletonize_3d(bin_im)
    pos_arr = np.argwhere(skel_im)
    pos = dict(zip(range(len(pos_arr)), pos_arr))
    nodes = set(pos)
    neighb = {}
    to_treat = set([min(nodes)])
    done = set()
    # Builds the tree of the skeleton
    # The nodes are the coordinates
    # Two nodes are linked iff they are 6-connected
    while 0 < len(to_treat):
        curr = to_treat.pop()
        done.add(curr)
        dist = np.abs(pos_arr[curr] - pos_arr)
        N = set(np.where(np.max(dist, axis=1)==1)[0])
        for ni in N:
            neighb.setdefault(curr, []).append(ni)
        to_treat.update(N.difference(done))

    # Finds the leaves of the tree
    extremities = [k for k, v in neighb.items() if len(v)==1]
    D_out = {}
    # For each leaf, finds the most distant leaf
    # Using Dijkstra algorithm
    with Pool() as pool:
        mapping = [(nodes, neighb, e) for e in extremities]
        out = pool.map(Dijkstra, mapping)
        pool.terminate()
        pool.close()
    # Finds the pair (e1, e2) of most distant leaves
    D_out = dict(zip(extremities, out))
    e1 = max(D_out, key=lambda x: D_out.get(x)[1])
    e2 = D_out[e1][0]
    prev = D_out[e1][2]
    curr = e2
    skel_im[skel_im!=0] = 1

    # Retrieve and smooth the longest path of the skeleton tree
    path = []
    while curr in prev:
        path += [pos[curr]]
        curr = prev[curr]
    X, Y, Z = zip(*path)
    X_smoothed = np.round(nd.filters.gaussian_filter1d(X, sigma=2)).astype(np.uint16)
    Y_smoothed = np.round(nd.filters.gaussian_filter1d(Y, sigma=2)).astype(np.uint16)
    Z_smoothed = np.round(nd.filters.gaussian_filter1d(Z, sigma=2)).astype(np.uint16)
    for x, y, z in zip(X_smoothed, Y_smoothed, Z_smoothed):
        skel_im[tuple([x, y, z])] = 2

    # Build the graph containing the longest path of the skeleton tree
    pos_arr = np.argwhere(skel_im==2)
    pos = dict(zip(range(len(pos_arr)), pos_arr))
    nodes = set(pos)
    neighb = {}
    to_treat = set([min(nodes)])
    done = set()
    while 0 < len(to_treat):
        curr = to_treat.pop()
        done.add(curr)
        dist = np.abs(pos_arr[curr] - pos_arr)
        N = set(np.where(np.max(dist, axis=1)==1)[0])
        for ni in N:
            neighb.setdefault(curr, []).append(ni)
        to_treat.update(N.difference(done))

    # Retrieve the x, y, z coordinates of the longest path
    first = list(neighb.keys())[0]
    last, b, prev = Dijkstra((nodes, neighb, first))
    last, b, prev = Dijkstra((nodes, neighb, last))
    current = last
    ordered_pos = [pos[current]]
    while prev.get(current, -1)!=-1:
        current = prev[current]
        ordered_pos += [pos[current]]
    x, y, z = zip(*ordered_pos)

    # Computes, smooth and interpolate the width along the longest path
    width = dist_trsf_im[(x, y, z)].flatten()
    width = nd.filters.gaussian_filter1d(width.astype(np.float), sigma=4)
    X = np.linspace(0, 1, len(width))
    width_spl = InterpolatedUnivariateSpline(X, width)
    # Computes the length of the longest path in physical units (given by vs)
    tmp = np.array(list(zip(x, y, z)))*vs
    length = np.sum(np.linalg.norm(tmp[:-1] - tmp[1:], axis=1))
    # Computes the Aspect Ration as the length over the median width
    AR = length/np.median(width)

    return (length, width_spl, AR, solidity, volume, surface, sphericity)

def compute_metrics2D(bin_im, dist_trsf_im, AP_pos=None, vs=None):
    """
    Given an isotropic 2D binary image `bin_im`, a distance transform image
    `dist_trsf_im` of `bin_im`, a 2D position and a voxel size `vs`,
    computes the length, the width, the aspect ratio, the solidity,
    the sphericity, the volume and the surface of the binary image.
    Notes: - `bin_im` has to be isotropic otherwise the distances won't be
           computed correctly
           - `bin_im` and `dist_trsf_im` must share the same shape

    Parameters
    ----------
    bin_im : array_like
        A n*m*l binary array with one single connected component
    dist_trsf_im : array_like
        A distance transformation of the bin_im array
    AP_pos : ((float, float), (float, float)) optional
        x, y position of the anterior part of the Neural Tube
    vs : float optional (default 1.)
        The size of the isotropic voxel in order to get measurements
        in physical units

    Returns
    -------
    length : float
        The length of the masked object in the binary image
    width_spl : scipy.interpolate.InterpolatedUnivariateSpline
        The interpolated width along the masked object
        in the binary image
    width_median : float
        Median of the width along the masked object in the binary image
    AR : float
        The aspect ratio of the masked object in the binary image
        `length/median(width)`
    solidity : float
        The solidity of the masked object in the binary image
        Ratio of pixels in the region to pixels of the convex hull image.
    surface :
        The surface of the masked object in the binary image
        Number of voxels in the region
    perimeter :
        The perimeter of the masked object in the binary image
    circularity :
        The circularity of the masked object in the binary image
        4*pi*(`surface`/(`perimeter`**2))
    """

    # Extract the largest connected component
    label_im = nd.label(bin_im)[0]
    labels = np.unique(label_im)
    labels = labels[labels!=0]
    surfaces = nd.sum(np.ones_like(label_im), index=labels, labels=label_im)
    final_cc = labels[np.argmax(surfaces)]
    bin_im = (label_im==final_cc).astype(np.uint8)

    # Compute the surface, perimeter, circularity and solidity
    if vs is None:
        vs = 1.
    props = measure.regionprops(bin_im.astype(np.uint8))[0]
    surface = props.area*vs**2
    perimeter = props.perimeter*vs
    circularity= 4*np.pi*(surface/perimeter**2)
    solidity = props.solidity

    # Skeletonize the mask and retrieve the coordinates of the skeleton
    skel_im = morphology.skeletonize(bin_im).astype(np.uint8)
    pos_arr = np.argwhere(skel_im)
    pos = dict(zip(range(len(pos_arr)), pos_arr))
    nodes = set(pos)
    neighb = {}
    to_treat = set([min(nodes)])
    done = set()
    # Builds the tree of the skeleton
    # The nodes are the coordinates
    # Two nodes are linked iff they are 4-connected
    while 0 < len(to_treat):
        curr = to_treat.pop()
        done.add(curr)
        dist = np.abs(pos_arr[curr] - pos_arr)
        N = set(np.where(np.max(dist, axis=1)==1)[0])
        for ni in N:
            neighb.setdefault(curr, []).append(ni)
        to_treat.update(N.difference(done))

    # Finds the leaves of the tree
    extremities = [k for k, v in neighb.items() if len(v)==1]
    D_out = {}
    # For each leaf, finds the most distant leaf
    # Using Dijkstra algorithm
    with Pool() as pool:
        mapping = [(nodes, neighb, e) for e in extremities]
        out = pool.map(Dijkstra, mapping)
        pool.terminate()
        pool.close()
    D_out = dict(zip(extremities, out))
    # Finds the pair (e1, e2) of most distant leaves
    e1 = max(D_out, key=lambda x: D_out.get(x)[1])
    e2 = D_out[e1][0]
    prev = D_out[e1][2]
    curr = e2

    # Retrieve and smooth the longest path of the skeleton tree
    skel_im[skel_im!=0] = 1
    path = []
    while curr in prev:
        path += [pos[curr]]
        curr = prev[curr]
    X, Y = zip(*path)
    X_smoothed = np.round(nd.filters.gaussian_filter1d(X, sigma=2)).astype(np.uint16)
    Y_smoothed = np.round(nd.filters.gaussian_filter1d(Y, sigma=2)).astype(np.uint16)
    for x, y in zip(X_smoothed, Y_smoothed):
        skel_im[tuple([x, y])] = 2

    # Build the graph containing the longest path of the skeleton tree
    pos_arr = np.argwhere(skel_im==2)
    pos = dict(zip(range(len(pos_arr)), pos_arr))
    nodes = set(pos)
    neighb = {}
    to_treat = set([min(nodes)])
    done = set()
    while 0 < len(to_treat):
        curr = to_treat.pop()
        done.add(curr)
        dist = np.abs(pos_arr[curr] - pos_arr)
        N = set(np.where(np.max(dist, axis=1)==1)[0])
        for ni in N:
            neighb.setdefault(curr, []).append(ni)
        to_treat.update(N.difference(done))

    # Retrieve the x, y, z coordinates of the longest path
    first = list(neighb.keys())[0]
    last, b, prev = Dijkstra((nodes, neighb, first))
    last, b, prev = Dijkstra((nodes, neighb, last))
    current = last
    ordered_pos = [pos[current]]
    while prev.get(current, -1)!=-1:
        current = prev[current]
        ordered_pos += [pos[current]]

    # Gets the closest points of the skeleton to
    # the manually informed posterior and anterior positions
    if AP_pos is not None:
        A_pos, P_pos = AP_pos
        A_pos = np.array(A_pos)
        P_pos = np.array(P_pos)
        dist_to_A = np.linalg.norm(ordered_pos-A_pos, axis=1)
        A_pos = np.argmin(dist_to_A)

        dist_to_P = np.linalg.norm(ordered_pos-P_pos, axis=1)
        P_pos = np.argmin(dist_to_P)

        # Crop and reorder (if necessary) the skeleton
        if A_pos<P_pos:
            ordered_pos = ordered_pos[A_pos:P_pos+1]
            ordered_pos = ordered_pos[::-1]
            P_pos, A_pos = A_pos, P_pos
        else:
            ordered_pos = ordered_pos[P_pos:A_pos+1]

    # Computes, smooth and interpolate the width along the longest path
    dist_trsf_im *= vs
    x, y = zip(*ordered_pos)
    width = dist_trsf_im[(x, y)].flatten()
    width = nd.filters.gaussian_filter1d(width.astype(np.float), sigma=4)
    X = np.linspace(0, 1, len(width))
    width_spl = InterpolatedUnivariateSpline(X, width)
    # Computes the length of the longest path in physical units (given by vs)
    tmp = np.array(list(zip(x, y)))*vs
    length = np.sum(np.linalg.norm(tmp[:-1] - tmp[1:], axis=1))
    # Computes the Aspect Ration as the length over the median width
    AR = length/np.median(width)

    return(length, width_spl, np.median(width), AR, solidity, surface, perimeter, circularity)