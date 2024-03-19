"""
Constants and helper functions for multiprocessing.
"""
from functools import partial
from multiprocessing import Pool, cpu_count, freeze_support
from multiprocessing.dummy import Pool as ThreadPool

from tqdm import tqdm

from ._typing import Callable, Iterable, Optional

### Thresholds - use multiprocessing when n > threshold
# File i/o
_THRESHOLD_LOADING_FILES = 1000
_CHUNKSIZE_LOADING_FILES = 96

# PCSE inputs
_THRESHOLD_SITE_GENERATION = 1000
_CHUNKSIZE_SITE_GENERATION = 96

# PCSE model
_THRESHOLD_RUNNING_PCSE = 200
_CHUNKSIZE_RUNNING_PCSE = 10


### Main functions
def multiprocess_if_over_threshold(func: Callable, data: Iterable, threshold: int, *,
                                   n: Optional[int]=None,
                                   use_threads: bool=False, chunksize: int=5,
                                   progressbar: bool=True, leave_progressbar: bool=False, desc: Optional[str]=None, unit: Optional[str]=None, tqdm_kwargs: Optional[dict]={},
                                   **kwargs) -> Iterable:
    """
    Apply the given `func` to the `data`.
    Use multiprocessing if there are more elements than the given threshold; otherwise use a simple mapping.
    If `use_threads` is False (default), a process Pool is used; if True, a ThreadPool is used.
    """
    ### Determine whether to use multiprocessing
    # If n was not provided explicitly, get it from the length of the data
    if n is None:
        try:
            n = len(data)
        except TypeError:  # e.g. if `data` is a generator
            n = None

    # See if n is over the threshold
    try:
        USE_MULTIPROCESSING = (n > threshold)
    except TypeError:  # If n is None, e.g. if `data` is a generator
        USE_MULTIPROCESSING = False

    ### Setup
    # Select the type of Pool to use
    _Pool = ThreadPool if use_threads else Pool

    # Set up a progressbar function
    # (multiprocessing doesn't always like running on a predefined tqdm object)
    _tqdm = partial(tqdm, total=n, desc=desc, unit=unit, disable=not progressbar, leave=leave_progressbar, **tqdm_kwargs)

    ### Apply the function
    if USE_MULTIPROCESSING:
        with _Pool() as p:
            outputs = list(_tqdm(p.imap_unordered(func, data, chunksize=chunksize)))
    else:
        outputs = list(map(func, _tqdm(data)))

    return outputs


### Convenience functions
multiprocess_file_io = partial(multiprocess_if_over_threshold, threshold=_THRESHOLD_LOADING_FILES, use_threads=False, chunksize=_CHUNKSIZE_LOADING_FILES, unit="file")

multiprocess_site_generation = partial(multiprocess_if_over_threshold, threshold=_THRESHOLD_SITE_GENERATION, use_threads=False, chunksize=_CHUNKSIZE_SITE_GENERATION, unit="site", desc="Generating sites")

multiprocess_pcse = partial(multiprocess_if_over_threshold, threshold=_THRESHOLD_RUNNING_PCSE, use_threads=False, chunksize=_CHUNKSIZE_RUNNING_PCSE, unit="run", desc="Running PCSE models")
