#!/usr/bin/env python
#--coding:utf-8--
"""
cmat.py
cLoops2 contact matrix and 1D signal pileup related functions.
2019-12-17: 1D track support added
"""

__author__ = "CAO Yaqiang"
__date__ = ""
__modified__ = ""
__email__ = "caoyaqiang0410@gmail.com"

#3rd
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix

#cLoops2
from cLoops2.ds import XY
from cLoops2.io import parseIxy


def xy2dict(mat, s=-1, e=-1, r=5000):
    """
    Convert the coordinates to contact matrix with specified resolution.
    @param mat: np.array, [[x,y]]
    @param s: start site to build the contact matrix, if -1, infer from the data min , assigned can help to maintain same shape
    @param e: end site to build the contact matrix, if -1, infer from the data max , assigned can help to maintain same shape
    @param r: resolution for the contact matrix
    """
    nmat = {}
    if s == -1:
        s = np.min(mat)
    if e == -1:
        e = np.max(mat)
    for x, y in mat:
        nx = int((x - s) / r)
        ny = int((y - s) / r)
        if nx not in nmat:
            nmat[nx] = {}
        if ny not in nmat[nx]:
            nmat[nx][ny] = 0
        nmat[nx][ny] += 1
    #add the max value to maintain the same shape of matrix across data
    nx = int((e - s) / r)
    ny = int((e - s) / r)
    if nx not in nmat or ny not in nmat[nx]:
        nmat[nx] = {ny: 0}
    return nmat


def ixy2dict(f, r=5000, cut=0):
    """
    Convert the .ixy file to contact matrix with specified resolution.
    @param f: .ixy file
    @param r: resolution for the contact matrix
    @param cut: distance cutoff to filter PETs
    """
    chrom, mat = parseIxy(f, cut=cut)
    return xy2dict(mat, r=r)


def dict2mat(nmat):
    """
    Conver contact matrix in dict to sparse matrix.
    @return scipy.sparse import csr_matrix
    """
    data, row, col = [], [], []
    for nx in nmat.keys():
        for ny in nmat[nx].keys():
            #create the symetric matrix
            data.append(nmat[nx][ny])
            row.append(nx)
            col.append(ny)
            data.append(nmat[nx][ny])
            row.append(ny)
            col.append(nx)
    cmat = csr_matrix((data, (row, col)))
    return cmat


def getObsMat(xy, start, end, r):
    """
    Get the observed interaction contact matrix.
    xy is [[x,y]]
    r is resolution
    """
    #ps = np.where(xy[:, 0] >= start)[0]
    #xy = xy[ps, ]
    #ps = np.where(xy[:, 1] <= end)[0]
    #xy = xy[ps, ]
    ps = np.where((xy[:, 0] >= start) & (xy[:, 1] <= end))[0]
    xy = xy[ps, ]
    mat = xy2dict(xy, s=start, e=end, r=r)
    mat = dict2mat(mat)
    mat = mat.toarray()
    return mat


def _validate_trans_matrix_args(xy, x_start, x_end, y_start, y_end,
                                x_bin_size, y_bin_size):
    """Validate inputs and return the shape of a closed-interval matrix."""
    xy = np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("trans contact coordinates must have shape (n, 2)")
    if xy.size > 0 and not np.issubdtype(xy.dtype, np.integer):
        raise TypeError("trans contact coordinates must be integers")
    values = (x_start, x_end, y_start, y_end, x_bin_size, y_bin_size)
    if any(isinstance(v, (bool, np.bool_)) or
           not isinstance(v, (int, np.integer)) for v in values):
        raise TypeError("matrix boundaries and bin sizes must be integers")
    x_start, x_end = int(x_start), int(x_end)
    y_start, y_end = int(y_start), int(y_end)
    x_bin_size, y_bin_size = int(x_bin_size), int(y_bin_size)
    if x_start > x_end or y_start > y_end:
        raise ValueError("matrix start must not exceed matrix end")
    if x_bin_size <= 0 or y_bin_size <= 0:
        raise ValueError("matrix bin sizes must be positive")
    # cLoops2 loop coordinates and range queries are inclusive.  Preserve
    # that convention explicitly: an endpoint exactly equal to ``*_end`` is
    # retained in the final bin.
    n_x = (x_end - x_start) // x_bin_size + 1
    n_y = (y_end - y_start) // y_bin_size + 1
    return (np.asarray(xy, dtype=np.int64), x_start, x_end, y_start, y_end,
            x_bin_size, y_bin_size, (n_x, n_y))


def getTransObsMatCOO(xy,
                      x_start,
                      x_end,
                      y_start,
                      y_end,
                      x_bin_size,
                      y_bin_size=None):
    """Build a non-symmetric sparse trans contact matrix.

    Rows are bins on ``chromX`` and columns are bins on ``chromY``.  Both
    coordinate ranges are closed intervals.  Unlike :func:`dict2mat`, this
    function never mirrors a contact into the transposed cell and therefore
    also never doubles an observation that happens to have equal numerical
    X/Y bin indices on different chromosomes.

    The function only materializes arrays proportional to the PET count and
    is suitable as the primitive for streaming/full-pair sparse output.
    """
    if y_bin_size is None:
        y_bin_size = x_bin_size
    (xy, x_start, x_end, y_start, y_end, x_bin_size, y_bin_size,
     shape) = _validate_trans_matrix_args(xy, x_start, x_end, y_start,
                                          y_end, x_bin_size, y_bin_size)
    if xy.shape[0] == 0:
        return coo_matrix(shape, dtype=np.int64)
    keep = ((xy[:, 0] >= x_start) & (xy[:, 0] <= x_end) &
            (xy[:, 1] >= y_start) & (xy[:, 1] <= y_end))
    selected = xy[keep]
    if selected.shape[0] == 0:
        return coo_matrix(shape, dtype=np.int64)
    rows = ((selected[:, 0] - x_start) // x_bin_size).astype(np.int64)
    cols = ((selected[:, 1] - y_start) // y_bin_size).astype(np.int64)
    data = np.ones(selected.shape[0], dtype=np.int64)
    # Converting through CSR sums duplicate PETs in the same cell while
    # retaining a compact canonical COO representation for serialization.
    return coo_matrix((data, (rows, cols)), shape=shape).tocsr().tocoo()


def getTransObsMat(xy,
                   x_start,
                   x_end,
                   y_start,
                   y_end,
                   x_bin_size,
                   y_bin_size=None,
                   max_dense_cells=10000000):
    """Build a bounded dense ``chromX bins × chromY bins`` matrix.

    This convenience function is intended for local plots and aggregate
    windows.  Full chromosome-pair output should use
    :func:`getTransObsMatCOO`.  The cell limit is checked before allocating
    the dense array.
    """
    if y_bin_size is None:
        y_bin_size = x_bin_size
    validated = _validate_trans_matrix_args(xy, x_start, x_end, y_start,
                                            y_end, x_bin_size, y_bin_size)
    shape = validated[-1]
    if (isinstance(max_dense_cells, (bool, np.bool_)) or
            not isinstance(max_dense_cells, (int, np.integer)) or
            max_dense_cells <= 0):
        raise ValueError("max_dense_cells must be a positive integer")
    cells = int(shape[0]) * int(shape[1])
    if cells > int(max_dense_cells):
        raise ValueError(
            "requested trans matrix has %s cells, exceeding the limit %s; "
            "use getTransObsMatCOO for a sparse matrix" %
            (cells, int(max_dense_cells)))
    return getTransObsMatCOO(xy, x_start, x_end, y_start, y_end,
                             x_bin_size, y_bin_size).toarray()


def getExpMat(xy, shape, start, end, r, repeats=5):
    """
    Get the expected interaction contact matrix.
    xy is [[x,y]]
    shape is () shape from the observed matrix.
    r is resolution
    """
    mat = []
    i = 0
    while i < repeats:
        a = xy[:, 0]
        b = xy[:, 1]
        np.random.shuffle(a)
        np.random.shuffle(b)
        xy[:, 0] = a
        xy[:, 1] = b
        s = b-a
        s = np.where( s > 0)[0]
        nxy = xy[s,] 
        nmat = getObsMat(nxy, start, end, r)
        if nmat.shape == shape:
            mat.append(nmat)
            i += 1
    mat = np.array(mat)
    return mat.mean(axis=0)


def get1DSig(xy, start, end, ext=50):
    """
    Get the overlayed 1D signal
    @param xy, cLoops2.ds.XY object
    @param start: int, start coordinate
    @param end: int, end coordinate
    @param ext: int, extention of each tag
    """
    ss = np.zeros(end - start)
    l_idx = np.searchsorted(xy.xs, start, side="left")
    r_idx = np.searchsorted(xy.xs, end, side="right")
    for i in range(l_idx, r_idx):
        x = xy.xs[i]
        pa = max(0, x - start - ext)
        pb = min(max(0, x - start + ext), end-start) #fix max
        ss[pa:pb] += 1
    l_idx = np.searchsorted(xy.ys, start, side="left")
    r_idx = np.searchsorted(xy.ys, end, side="right")
    for i in range(l_idx, r_idx):
        y = xy.ys[i]
        pa = max(0, y - start - ext)
        pb = min(max(0, y - start + ext), end)
        ss[pa:pb] += 1
    return ss


def get1DSigPE(xy, start, end, ext=50):
    """
    Get the overlayed 1D signal for paired-end tags.
    @param xy, cLoops2.ds.XY object
    @param start: int, start coordinate
    @param end: int, end coordinate
    @param ext: int, extention of each tag
    """
    ss = np.zeros(end - start)
    ps = list(xy.queryPeakBoth(start, end))
    for p in ps:
        x = xy.mat[p,0]
        y = xy.mat[p,1]
        pa = max(0, x - start - ext)
        pb = min(max(0, y - start + ext), end)
        ss[pa:pb] += 1
    return ss


def getBinMean(s, bins=100):
    """
    Get the mean of bins for a array.
    @param s: np.array
    @param bins: int, how many bins as converted
    """
    width = int(len(s) / bins)
    ns = s[:bins * width].reshape(-1, width).mean(axis=1)
    return ns


def get1DSigMat(xy, rs, ext=20, bins=100, skipZeros=False):
    """
    Get the 1D signal matrix for a set of regions.
    @param xy is XY object
    ext is extend of reads from the center
    bins is the final array size for a record
    return a pd.Dataframe, row is regions/peaks, columns is j
    """
    ds = {}
    print("Get 1D signal for %s regions" % (len(rs)))
    for r in tqdm(rs):
        #for r in rs:
        if r[2] - r[1] < bins:
            continue
        s = get1DSig(xy, int(r[1]), int(r[2]), ext=ext)
        if skipZeros and np.sum(s) == 0:
            continue
        ns = getBinMean(s, bins=bins)
        if len(r) >= 4:
            rid = "|".join(list(map(str, r[:4])))
        else:
            rid = "|".join(list(map(str, r[:3])))
        ds[rid] = ns
    if len(ds) == 0:
        return None
    else:
        ds = pd.DataFrame(ds).T
        return ds



def getVirtual4CSig(xy,start,end,viewStart,viewEnd,ext=20):
    """
    Get the virtual 4C signal for a region and a view point.
    @param xy, cLoops2.ds.XY object
    @param start: int, start coordinate
    @param end: int, end coordinate
    @param viewStart: int, start coordinate for view point
    @param viewEnd: int, end coordinate for view point
    """
    aps = xy.queryPeak(start,end)
    bps = xy.queryPeak(viewStart,viewEnd)
    cps = xy.queryPeakBoth(viewStart,viewEnd)
    ps = aps.intersection( bps ).difference( cps )
    ss = np.zeros(end - start)
    for i in ps:
        x = xy.mat[i,0]
        y = xy.mat[i,1]
        pa = max(0, x - start - ext)
        pb = min(max(0, x - start + ext), end-start) #fix max
        ss[pa:pb] += 1
        pa = max(0, y - start - ext)
        pb = min(max(0, y - start + ext), end)
        ss[pa:pb] += 1
    return ss


