#!/usr/bin/env python
#--coding:utf-8--
"""
ds.py
Defined data structure used in cLoops2.

2020-04-20: update the xy.queryLoop, changed to old way, preiviouse one (lefta,leftb, righta,righb), if rightb < lefta, will call 0
2021-04-01: add summit for peak
2021-05-20: add mat attribute to XY object for raw data access
"""

__author__ = "CAO Yaqiang"
__email__ = "caoyaqiang0410@gmail.com"

import numpy as np


class PET(object):
    """
    Paired-end tags / PETs object.
    """
    __slots__ = [
        "chromA",
        "chromB",
        "startA",
        "startB",
        "endA",
        "endB",
        "strandA",
        "strandB",
        "cA",
        "cB",
        "distance",
        "cis",
        "length",
        "mid",
        "mapq",
        "start",
        "end",
    ]

    def __init__(self, d):
        """
        d is line = line.split( "\n" )[ 0 ].split( "\t" ) from BEDPE file 
        """
        self.chromA = d[0]
        self.startA = int(d[1])
        self.endA = int(d[2])
        self.strandA = d[8]
        self.chromB = d[3]
        self.startB = int(d[4])
        self.endB = int(d[5])
        self.strandB = d[9]
        try:
            self.mapq = int(d[7])
        except:
            self.mapq = 255  #if no mapq information available, all set high
        if self.chromA == self.chromB:
            self.cis = True
            #adjust the left end and right end to make sure left is alwasy small than right
            if self.startA + self.endA > self.startB + self.endB:
                self.startA, self.startB = self.startB, self.startA
                self.endA, self.endB = self.endB, self.endA
                self.strandA, self.strandB = self.strandB, self.strandA
            self.cA = int((self.startA + self.endA) / 2)
            self.cB = int((self.startB + self.endB) / 2)
            self.distance = int(abs(self.cB -
                                    self.cA))  #used for long-range PETs
            self.length = int(
                self.endB -
                self.startA)  #fragment length, used for short-range PETs
            self.mid = int(
                (self.startA + self.endB) / 2)  #middle of the fragment
            self.start = self.startA
            self.end = self.endB
        else:
            self.cis = False
            self.distance = None
            self.length = None
            self.mid = None
            #adjust the left end and right end to make sure left is alwasy small than right, (chr1,chr2)
            if self.chromA > self.chromB:
                self.chromA, self.chromB = self.chromB, self.chromA
                self.startA, self.startB = self.startB, self.startA
                self.endA, self.endB = self.endB, self.endA
                self.strandA, self.strandB = self.strandB, self.strandA
            self.cA = int((self.startA + self.endA) / 2)
            self.cB = int((self.startB + self.endB) / 2)



class Pair(object):
    """
    Pair object.
    Data format according to:https://pairtools.readthedocs.io/en/latest/formats.html#pairs
    """
    __slots__ = [
        "chromA",
        "chromB",
        "cA",
        "cB",
        "strandA",
        "strandB",
        "distance",
        "cis",
        "mid",
        "start",
        "end",
    ]

    def __init__(self, d):
        """
        d is line = line.split( "\n" )[ 0 ].split( "\t" ) from BEDPE file 
        """
        self.chromA = d[1]
        self.cA = int( d[2] )
        self.chromB = d[3]
        self.cB = int( d[4] )
        self.strandA = d[5]
        self.strandB = d[6]
        if self.chromA == self.chromB:
            self.cis = True
            #adjust the left end and right end to make sure left is alwasy small than right
            if self.cA > self.cB:
                self.cA, self.cB = self.cB, self.cA
                self.strandA, self.strandB = self.strandB, self.strandA
            self.distance = int(abs(self.cB - self.cA)) 
            self.mid = int( (self.cA + self.cB) / 2)  #middle of the fragment
        else:
            self.cis = False
            self.distance = None
            self.mid = None
            #adjust the left end and right end to make sure left is alwasy small than right, (chr1,chr2)
            if self.chromA > self.chromB:
                self.chromA, self.chromB = self.chromB, self.chromA
                self.cA, self.cB = self.cB, self.cA
                self.strandA, self.strandB = self.strandB, self.strandA



class XY(object):
    """
    x,y coordinates for fast access, query point numbers and ids.
    """

    def __init__(self, xs, ys):
        """
        xs: [1,2,3]
        ys: [4,5,6]
        xs and ys should be the same length.
        (x,y) is the locatation for a PET.
        """
        self.number = len(xs)
        x2i, y2i = {}, {}
        for i, x in enumerate(xs):
            x2i.setdefault(x, []).append(i)
        for i, y in enumerate(ys):
            y2i.setdefault(y, []).append(i)
        self.mat = np.array( [[xs[i],ys[i]] for i in range(len(xs)) ] ) #orgianl 
        self.xs = np.sort(np.array(xs)) #sorted xs
        self.ys = np.sort(np.array(ys)) #sorted ys
        self.x2i = x2i #original index for values
        self.y2i = y2i

    def _query(self, cor, cor2i, left, right):
        """
        For a sorted one-dimension numpy array, query the points id in a region.
        """
        ps = set()
        l_idx = np.searchsorted(cor, left, side="left")
        r_idx = np.searchsorted(cor, right, side="right")
        for i in range(l_idx, r_idx):
            ps.update(cor2i[cor[i]])
        return ps

    def queryPeak(self, left, right):
        """
        Get the all index for points in a region, only one end is enough.
        """
        xps = self._query(self.xs, self.x2i, left, right)
        yps = self._query(self.ys, self.y2i, left, right)
        return xps.union(yps)

    def queryPeakBoth(self, left, right):
        """
        Get the PETs that both ends with in the peak.
        """
        xps = self._query(self.xs, self.x2i, left, right)
        yps = self._query(self.ys, self.y2i, left, right)
        return xps.intersection(yps)

    def queryLoop(self, lefta, righta, leftb, rightb):
        """
        Get the all index for points in two linked regions.
        """
        aps = self.queryPeak(lefta, righta)
        bps = self.queryPeak(leftb, rightb)
        return aps, bps, aps.intersection(bps)
        """
        #for this method, righta should always <= leftb
        sourcea = self._query(self.xs, self.x2i, lefta, righta)
        targeta = self._query(self.ys, self.y2i, lefta, righta)
        sourceb = self._query(self.xs, self.x2i, leftb, rightb)
        targetb = self._query(self.ys, self.y2i, leftb, rightb)
        aps = sourcea.union(targeta)
        bps = sourceb.union(targetb)
        return aps, bps, sourcea.intersection(targetb)
        """


class TransContactIndex(object):
    """Axis-aware index for one inter-chromosomal ``.ixy`` matrix.

    A trans ``.ixy`` record is directional: column 0 belongs to ``chromX``
    and column 1 belongs to ``chromY``.  Unlike :class:`XY`, this class never
    treats the two columns as interchangeable.  Query intervals are closed
    (``left <= coordinate <= right``), matching the historical ``XY`` query
    convention used by cLoops2.

    The input coordinates must be one-dimensional integer arrays with equal
    length.  Empty inputs are accepted.  Stable sorted indexes make ties
    deterministic while the original coordinate arrays remain unchanged.
    Construction is ``O(N log N)``; an axis count is ``O(log N)`` and a
    rectangle query is ``O(log N + min(kx, ky))`` after candidate filtering.
    """

    def __init__(self, xs, ys):
        self._x = self._as_coordinates(xs, "xs")
        self._y = self._as_coordinates(ys, "ys")
        if self._x.shape[0] != self._y.shape[0]:
            raise ValueError("xs and ys must contain the same number of PETs")

        self.number = int(self._x.shape[0])
        self._x_order = np.argsort(self._x, kind="mergesort")
        self._y_order = np.argsort(self._y, kind="mergesort")
        self._x_sorted = self._x[self._x_order]
        self._y_sorted = self._y[self._y_order]

    @staticmethod
    def _as_coordinates(values, name):
        """Validate and copy one coordinate axis as an ``int64`` array."""
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError("%s must be a one-dimensional array" % name)
        # ``np.asarray([])`` has float dtype although there are no values to
        # validate.  Treat an empty axis as a valid empty integer coordinate
        # array, but reject non-integer populated inputs rather than silently
        # truncating coordinates.
        if values.size == 0:
            return np.asarray(values, dtype=np.int64).copy()
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("%s must contain integer coordinates" % name)
        limits = np.iinfo(np.int64)
        if np.any(values < limits.min) or np.any(values > limits.max):
            raise OverflowError("%s coordinates exceed int64 range" % name)
        return np.asarray(values, dtype=np.int64).copy()

    @staticmethod
    def _validate_interval(left, right):
        if isinstance(left, (bool, np.bool_)) or not isinstance(
                left, (int, np.integer)):
            raise TypeError("interval boundaries must be integers")
        if isinstance(right, (bool, np.bool_)) or not isinstance(
                right, (int, np.integer)):
            raise TypeError("interval boundaries must be integers")
        left, right = int(left), int(right)
        if left > right:
            raise ValueError("interval left boundary must not exceed right")
        return left, right

    @classmethod
    def from_matrix(cls, mat):
        """Build an index from an integer matrix with exactly two columns."""
        mat = np.asarray(mat)
        if mat.ndim != 2 or mat.shape[1] != 2:
            raise ValueError("trans contact matrix must have shape (n, 2)")
        if mat.size > 0 and not np.issubdtype(mat.dtype, np.integer):
            raise TypeError("trans contact matrix must contain integers")
        return cls(mat[:, 0], mat[:, 1])

    @staticmethod
    def _axis_bounds(sorted_values, left, right):
        left_i = np.searchsorted(sorted_values, left, side="left")
        right_i = np.searchsorted(sorted_values, right, side="right")
        return int(left_i), int(right_i)

    def _axis_ids(self, axis, left, right):
        left, right = self._validate_interval(left, right)
        if axis == "x":
            values, order = self._x_sorted, self._x_order
        elif axis == "y":
            values, order = self._y_sorted, self._y_order
        else:
            raise ValueError("axis must be 'x' or 'y'")
        left_i, right_i = self._axis_bounds(values, left, right)
        return order[left_i:right_i]

    def count_x(self, left, right):
        """Count PETs whose ``chromX`` coordinate is in a closed interval."""
        left, right = self._validate_interval(left, right)
        left_i, right_i = self._axis_bounds(self._x_sorted, left, right)
        return right_i - left_i

    def count_y(self, left, right):
        """Count PETs whose ``chromY`` coordinate is in a closed interval."""
        left, right = self._validate_interval(left, right)
        left_i, right_i = self._axis_bounds(self._y_sorted, left, right)
        return right_i - left_i

    def query_rect_ids(self, x_left, x_right, y_left, y_right):
        """Return original PET ids inside a directional X-by-Y rectangle.

        The less populated one-dimensional query is used as the candidate
        set, then filtered on the other axis.  Returned ids are sorted by
        original row number, making results deterministic and convenient for
        direct matrix indexing.
        """
        x_left, x_right = self._validate_interval(x_left, x_right)
        y_left, y_right = self._validate_interval(y_left, y_right)
        x_ids = self._axis_ids("x", x_left, x_right)
        y_ids = self._axis_ids("y", y_left, y_right)

        if x_ids.size <= y_ids.size:
            candidates = x_ids
            other = self._y[candidates]
            keep = (other >= y_left) & (other <= y_right)
        else:
            candidates = y_ids
            other = self._x[candidates]
            keep = (other >= x_left) & (other <= x_right)
        return np.sort(candidates[keep], kind="mergesort")

    def count_rect(self, x_left, x_right, y_left, y_right):
        """Count PETs inside a directional X-by-Y rectangle."""
        return int(
            self.query_rect_ids(x_left, x_right, y_left, y_right).shape[0])



class Peak(object):
    """
    Used to store peak related information.
    """
    __slots__ = [
        "id",
        "chrom",
        "start",
        "end",
        "summit",
        "counts",
        "length",
        "density",  #RPKM
        "poisson_p_value",
        "enrichment_score",
        "control_counts",
        "control_local_counts",
        "control_density",
        "control_scaled_counts",
        "poisson_p_value_vs_control",
        "enrichment_score_vs_control",
        'up_down_counts',
        'control_up_down_counts', #direct up-stream and down-stream same size window
        "p_value_mean",
        "significant",
    ]

    def __str__(self):
        return str(self.__class__) + ": <%s:%s-%s>" % (self.chrom, self.start,
                                                       self.end)



class Loop(object):
    """
    Used to store loop related information.
    """
    __slots__ = [
        "id",
        "chromX",
        "chromY",
        "x_start",
        "x_end",
        "x_center",
        "ra",
        "y_start",
        "y_end",
        "y_center",
        "rb",
        "rab",
        "cis",
        "distance",
        "density",
        "ES",
        "P2LL",
        "FDR",
        "hypergeometric_p_value",
        "poisson_p_value",
        "binomial_p_value",
        "x_peak_poisson_p_value",
        "x_peak_es",
        "y_peak_poisson_p_value",
        "y_peak_es",
        "significant",
    ]

    def __str__(self):
        return str(self.__class__) + ": <%s:%s-%s><%s:%s-%s>" % (
            self.chromX, self.x_start, self.x_end, self.chromY, self.y_start,
            self.y_end)



class DiffLoop(object):
    """
    Used to store differentially enriched loop related information.
    """
    __slots__ = [
        "id",
        "chromX",
        "chromY",
        "x_start",
        "x_end",
        "x_center",
        "y_start",
        "y_end",
        "y_center",
        "distance",
        "size", #total length of the two anchors
        "raw_trt_ra",
        "raw_trt_rb",
        "raw_con_ra",
        "raw_con_rb",
        "scaled_trt_ra",
        "scaled_trt_rb",
        "raw_trt_rab",  #raw counts in target sample
        "raw_con_rab",
        "raw_trt_mrab",  #raw mean nearby counts in target
        "raw_con_mrab",
        "scaled_trt_rab",
        "scaled_trt_mrab",
        "trt_density",  #interaction density
        "con_density",
        "trt_es", #enrichment score compared to nearby region
        "con_es",
        "raw_fc",  #log2 transformed fold change
        "scaled_fc",  #log2 transformed fold change
        "poisson_p_value",
        "significant",
    ]

    def __str__(self):
        return str(self.__class__) + ": <%s:%s-%s><%s:%s-%s>" % (
            self.chromX, self.x_start, self.x_end, self.chromY, self.y_start,
            self.y_end)



class Domain(object):
    """
    Used to store peak related information.
    """
    __slots__ = [
        "id",
        "chrom",
        "start",
        "end",
        "counts",
        "length",
        "bs",  #bin size for the score
        "ws", #window size for the core
        "ss",  #correlation mean segragation score
        "totalPETs",
        "withinDomainPETs",
        "enrichmentScore",
        "density", #similar to RPKM,
        "significant",
    ]

    def __str__(self):
        return str(self.__class__) + ": <%s:%s-%s>" % (self.chrom, self.start,
                                                       self.end)



class Exon(object):
    __slots__ = [
        "chrom",
        "start",
        "end",
        "length",
        "strand",
        "name",
        "id",
    ]
    def __str__(self):
        return str(self.__class__) + ": <%s:%s-%s;%s;Exon;%s>" % (
                                                    self.chrom,
                                                    self.start,
                                                    self.end,
                                                    self.strand,
                                                    self.name,
                                                    )




class Gene(object):
    """
    Gene or transcript.
    """
    __slots__ = [
        "chrom",
        "start",
        "end",
        "length",
        "strand",
        "name",
        "id",
        "exons",
    ]
    def __str__(self):
        return str(self.__class__) + ": <%s:%s-%s;%s;Gene;%s>" % (
                                                    self.chrom,
                                                    self.start,
                                                    self.end,
                                                    self.strand,
                                                    self.name,
                                                    )
