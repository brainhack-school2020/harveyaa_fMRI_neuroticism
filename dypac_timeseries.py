# Chopped up to use with preprocessed time series by annabelle harvey
"""Dynamic Parcel Aggregation with Clustering (dypac)."""

# Authors: Pierre Bellec, Amal Boukhdir
# License: BSD 3 clause
import warnings

from scipy.sparse import vstack
import numpy as np

from sklearn.utils import check_random_state

from joblib import Memory
from nilearn.decomposition.base import BaseDecomposition

class Dypac(BaseDecomposition):
    """
    Perform Stable Dynamic Cluster Analysis.

    Parameters
    ----------
    n_clusters: int, optional
        Number of clusters to extract per time window

    n_states: int, optional
        Number of expected dynamic states

    n_replications: int, optional
        Number of replications of cluster analysis in each fMRI run

    n_batch: int, optional
        Number of batches to run through consensus clustering.
        If n_batch<=1, consensus clustering will be applied
        to all replications in one pass. Processing with batch will
        reduce dramatically the compute time, but will change slightly
        the results.

    n_init: int, optional
        Number of initializations for k-means

    subsample_size: int, optional
        Number of time points in a subsample

    max_iter: int, optional
        Max number of iterations for k-means

    threshold_sim: float (0 <= . <= 1), optional
        Minimal acceptable average dice in a state

    random_state: int or RandomState, optional
        Pseudo number generator state used for random sampling.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.

    detrend: boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    memory: instance of joblib.Memory or str
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    verbose: integer, optional
        Indicate the level of verbosity. By default, print progress.

    """

    def __init__(
        self,
        n_clusters=10,
        n_states=3,
        n_replications=40,
        n_batch=1,
        n_init=30,
        n_init_aggregation=100,
        subsample_size=30,
        max_iter=30,
        threshold_sim=0.3,
        random_state=None,
        standardize=True,
        detrend=True,
        low_pass=None,
        high_pass=None,
        t_r=None,
        memory=Memory(cachedir=None),
        memory_level=0,
        verbose=1,
    ):
        """Set up default attributes for the class."""
        # All those settings are taken from nilearn BaseDecomposition
        self.random_state = random_state
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory = memory
        self.memory_level = max(0, memory_level + 1)
        self.verbose = verbose

        # Those settings are specific to parcel aggregation
        self.n_clusters = n_clusters
        self.n_states = n_states
        self.n_batch = n_batch
        self.n_replications = n_replications
        self.n_init = n_init
        self.n_init_aggregation = n_init_aggregation
        self.subsample_size = subsample_size
        self.max_iter = max_iter
        self.threshold_sim = threshold_sim

    def _check_components_(self):
        """Check for presence of estimated components."""
        if not hasattr(self, "components_"):
            raise ValueError(
                "Object has no components_ attribute. "
                "This is probably because fit has not "
                "been called."
            )

    def fit(self, series):
        """
        Compute the mask and the dynamic parcels across datasets.

        Parameters
        ----------
        series: list of 4800x200 timeseries (timepoints x nodes)
 
         Returns
         -------
         self: object
            Returns the instance itself. Contains attributes listed
            at the object level.
        """

        # Control random number generation
        self.random_state = check_random_state(self.random_state)
        
        # Check that number of batches is reasonable
        if self.n_batch > len(series):
            warnings.warn(
                "{0} batches were requested, but only {1} datasets available. Using {2} batches instead.".format(
                    self.n_batch, len(series), self.n_batch
                )
            )
            self.n_batch = len(series)

        # reduce step
        if self.n_batch > 1:
            stab_maps, dwell_time = self._reduce_batch(series)
        else:
            stab_maps, dwell_time = self._reduce(series)

        # Return components
        self.components_ = stab_maps
        self.dwell_time_ = dwell_time

        # Create embedding
        self.embedding = Embedding(stab_maps.todense())
        return self

    def _reduce_batch(self, all_series):
        """Iterate dypac on batches of files."""
        stab_maps_list = []
        dwell_time_list = []
        for bb in range(self.n_batch):
            slice_batch = slice(bb, len(all_series), self.n_batch)
            if self.verbose:
                print("[{0}] Processing batch {1}".format(self.__class__.__name__, bb))
            stab_maps, dwell_time = self._reduce(
                all_series[slice_batch]
            )
            stab_maps_list.append(stab_maps)
            dwell_time_list.append(dwell_time)

        stab_maps_cons, dwell_time_cons = consensus_batch(
            stab_maps_list,
            dwell_time_list,
            self.n_replications,
            self.n_states,
            self.max_iter,
            self.n_init_aggregation,
            self.random_state,
            self.verbose,
        )

        return stab_maps_cons, dwell_time_cons

    def _reduce(self, all_series):
        """
        Cluster aggregation on a list of 4800x200 timeseries (timepoints x nodes)

        Returns
        -------
        stab_maps: ndarray
            stability maps of each state.

        dwell_time: ndarray
            dwell time of each state.
        """
        onehot_list = []
        for ind, time_series in zip(range(len(all_series)), all_series):
            
            onehot = replicate_clusters(
                time_series.transpose(),
                subsample_size=self.subsample_size,
                n_clusters=self.n_clusters,
                n_replications=self.n_replications,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
                desc="Replicating clusters in data #{0}".format(ind),
                verbose=self.verbose,
            )
            onehot_list.append(onehot)
        onehot_all = vstack(onehot_list)
        del onehot_list
        del onehot

        # find the states
        states = find_states(
            onehot_all,
            n_states=self.n_states,
            max_iter=self.max_iter,
            threshold_sim=self.threshold_sim,
            random_state=self.random_state,
            n_init=self.n_init_aggregation,
            verbose=self.verbose,
        )

        # Generate the stability maps
        stab_maps, dwell_time = produce_stab_maps(
            onehot_all, states, self.n_replications, self.n_states
        )

        return stab_maps, dwell_time
    
"""
Bagging analysis of stable clusters (BASC)++.
Scalable and fast ensemble clustering.
"""

# Authors: Pierre Bellec, Amal Boukhdir
# License: BSD 3 clause
import warnings
from tqdm import tqdm

from scipy.sparse import csr_matrix, find, vstack

from sklearn.cluster import k_means
from sklearn.preprocessing import scale


def _select_subsample(y, subsample_size, start=None):
    """Select a random subsample in a data array."""
    n_samples = y.shape[1]
    subsample_size = np.min([subsample_size, n_samples])
    max_start = n_samples - subsample_size
    if start is not None:
        start = np.min([start, max_start])
    else:
        start = np.floor((max_start + 1) * np.random.rand(1))
    stop = start + subsample_size
    samp = y[:, np.arange(int(start), int(stop))]
    return samp


def _part2onehot(part, n_clusters=0):
    """
    Convert a series of partition (one per row) with integer clusters into
    a series of one-hot encoding vectors (one per row and cluster).
    """
    if n_clusters == 0:
        n_clusters = np.max(part) + 1
    n_part, n_voxel = part.shape
    n_el = n_part * n_voxel
    val = np.repeat(True, n_el)
    ind_r = np.reshape(part, n_el) + np.repeat(
        np.array(range(n_part)) * n_clusters, n_voxel
    )
    ind_c = np.repeat(
        np.reshape(range(n_voxel), [1, n_voxel]), n_part, axis=0
    ).flatten()
    s_onehot = [n_part * n_clusters, n_voxel]
    onehot = csr_matrix((val, (ind_r, ind_c)), shape=s_onehot, dtype="bool")
    return onehot


def _start_window(n_time, n_replications, subsample_size):
    """Get a list of the starting points of sliding windows."""
    max_replications = n_time - subsample_size + 1
    n_replications = np.min([max_replications, n_replications])
    list_start = np.linspace(0, max_replications, n_replications)
    list_start = np.floor(list_start)
    list_start = np.unique(list_start)
    return list_start


def _trim_states(onehot, states, n_states, verbose, threshold_sim):
    """Trim the states clusters to exclude outliers."""
    for ss in tqdm(range(n_states), disable=not verbose, desc="Trimming states"):
        ix, iy, _ = find(onehot[states == ss, :])
        size_onehot = np.array(onehot[states == ss, :].sum(axis=1)).flatten()
        ref_cluster = np.array(onehot[states == ss, :].mean(dtype="float", axis=0))
        avg_stab = np.divide(
            np.bincount(ix, weights=ref_cluster[0, iy].flatten()), size_onehot
        )
        tmp = states[states == ss]
        tmp[avg_stab < threshold_sim] = n_states
        states[states == ss] = tmp
    return states


def replicate_clusters(
    y,
    subsample_size,
    n_clusters,
    n_replications,
    max_iter=100,
    n_init=10,
    random_state=None,
    verbose=False,
    embedding=np.array([]),
    desc="",
    normalize=False,
):
    """
    Replicate a clustering on random subsamples.

    Parameters
    ----------
    y: numpy array
        size number of samples x number of features

    subsample_size: int
        The size of the subsample used to generate cluster replications

    n_clusters: int
        The number of clusters to be extracted by k-means.

    n_replications: int
        The number of replications

    n_init: int, optional
            Number of initializations for k-means

    max_iter: int, optional
        Max number of iterations for the k-means algorithm

    verbose: boolean, optional
        Turn on/off verbose

    embedding: array, optional
        if present, the embedding array will be appended to samp for each sample.
        For example, embedding can be a set of spatial coordinates,
        to encourage spatial proximity in the clusters.

    desc: string, optional
        message to insert in verbose

    normalize: boolean, optional
        turn on/off scaling of each sample to zero mean and unit variance

    Returns
    -------
    onehot: boolean, sparse array
        onehot representation of clusters, stacked over all replications.
    """
    list_start = _start_window(y.shape[1], n_replications, subsample_size)
    if list_start.shape[0] < n_replications:
        warnings.warn(
            "{0} replications were requested, but only {1} available.".format(
                n_replications, list_start.shape[0]
            )
        )
    range_replication = range(list_start.shape[0])
    part = np.zeros([list_start.shape[0], y.shape[0]], dtype="int")

    for rr in tqdm(range_replication, disable=not verbose, desc=desc):
        if normalize:
            samp = scale(_select_subsample(y, subsample_size, list_start[rr]), axis=1)
        else:
            samp = _select_subsample(y, subsample_size, list_start[rr])
        if embedding.shape[0] > 0:
            samp = np.concatenate(
                [_select_subsample(y, subsample_size, list_start[rr]), embedding],
                axis=1,
            )
        _, part[rr, :], _ = k_means(
            samp,
            n_clusters=n_clusters,
            init="k-means++",
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
    return _part2onehot(part, n_clusters)


def find_states(
    onehot,
    n_states=10,
    max_iter=30,
    threshold_sim=0.3,
    n_init=10,
    random_state=None,
    verbose=False,
):
    """Find dynamic states based on the similarity of clusters over time."""
    if verbose:
        print("Consensus clustering.")
    _, states, _ = k_means(
        onehot,
        n_clusters=n_states,
        init="k-means++",
        max_iter=max_iter,
        random_state=random_state,
        n_init=n_init,
    )
    states = _trim_states(onehot, states, n_states, verbose, threshold_sim)
    return states


def produce_stab_maps(onehot, states, n_replications, n_states, dwell_time_all=None):
    """Generate stability maps associated with different states."""
    # Dwell times
    dwell_time = np.zeros(n_states)
    for ss in range(0, n_states):
        if np.any(dwell_time_all == None):
            dwell_time[ss] = np.sum(states == ss) / n_replications
        else:
            dwell_time[ss] = np.mean(dwell_time_all[states == ss])
    # Re-order stab maps by descending dwell time
    indsort = np.argsort(-dwell_time)
    dwell_time = dwell_time[indsort]

    # Stability maps
    row_ind = []
    col_ind = []
    val = []
    for idx, ss in enumerate(indsort):
        if np.any(states == ss):
            stab_map = onehot[states == ss, :].mean(dtype="float", axis=0)
            mask = stab_map > 0

            row_ind.append(np.repeat(idx, np.sum(mask)))
            col_ind.append(np.nonzero(mask)[1])
            val.append(np.array(stab_map[mask]).flatten())
    stab_maps = csr_matrix(
        (np.concatenate(val), (np.concatenate(row_ind), np.concatenate(col_ind))),
        shape=[n_states, onehot.shape[1]],
    )

    return stab_maps, dwell_time


def consensus_batch(
    stab_maps_list,
    dwell_time_list,
    n_replications,
    n_states=10,
    max_iter=30,
    n_init=10,
    random_state=None,
    verbose=False,
):
    stab_maps_all = vstack(stab_maps_list)
    del stab_maps_list
    dwell_time_all = np.concatenate(dwell_time_list)
    del dwell_time_list

    # Consensus clustering step
    if verbose:
        print("Inter-batch consensus")
    _, states_all, _ = k_means(
        stab_maps_all,
        n_clusters=n_states,
        sample_weight=dwell_time_all,
        init="k-means++",
        max_iter=max_iter,
        random_state=random_state,
        n_init=n_init,
    )

    # average stability maps and dwell times across consensus states
    if verbose:
        print("Generating consensus stability maps")
    stab_maps_cons, dwell_time_cons = produce_stab_maps(
        stab_maps_all, states_all, n_replications, n_states, dwell_time_all
    )
    return stab_maps_cons, dwell_time_cons

from sklearn.preprocessing import StandardScaler


def miss_constant(X, precision=1e-10):
    """Check if a constant vector is missing in a vector basis.
    """
    return np.min(np.sum(np.absolute(X - 1), axis=1)) > precision


class Embedding:
    def __init__(self, X, add_constant=True):
        """
        Transformation to and from an embedding.

        Parameters
        ----------
        X: ndarray
            The vector basis defining the embedding (each row is a vector).

        add_constant: boolean
            Add a constant vector to the vector basis, if none is present.

        Attributes
        ----------
        size: int
            the number of vectors defining the embedding, not including the
            intercept.

        transform_mat: ndarray
            matrix projection from original to embedding space.

        inverse_transform_mat: ndarray
            matrix projection from embedding to original space.

        """
        self.size = X.shape[0]
        # Once we have the embedded representation beta, the inverse transform
        # is a simple linear mixture:
        # Y_hat = beta * X
        # We store X as the inverse transform matrix
        if add_constant and miss_constant(X):
            self.inverse_transform_mat = np.concatenate([np.ones([1, X.shape[1]]), X])
        else:
            self.inverse_transform_mat = X
        # The embedded representation beta is also derived by a simple linear
        # mixture Y * P, where P is the pseudo-inverse of X
        # We store P as our transform matrix
        self.transform_mat = np.linalg.pinv(self.inverse_transform_mat)

    def transform(self, data):
        """Project data in embedding space."""
        # Given Y, we get
        # beta = Y * P
        return np.matmul(data, self.transform_mat)

    def inverse_transform(self, embedded_data):
        """Project embedded data back to original space."""
        # Given beta, we get:
        # Y_hat = beta * X
        return np.matmul(embedded_data, self.inverse_transform_mat)

    def compress(self, data):
        """Embedding compression of data in original space."""
        # Given Y, by combining transform and inverse_transform, we get:
        # Y_hat = Y * P * X
        return self.inverse_transform(self.transform(data))

    def score(self, data):
        """Average residual squares after compress in embedding space."""
        # The R2 score is only interpretable for standardized data
        data = StandardScaler().fit_transform(data)
        return 1 - np.var(data - self.compress(data), axis=0)