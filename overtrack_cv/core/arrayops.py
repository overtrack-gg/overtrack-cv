import bisect
from collections import Counter
from typing import Iterable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

Num = Union[int, float]


def argmin(seq: Union[Sequence[Num], np.ndarray]) -> int:
    return int(np.argmin(seq))


def argmax(seq: Union[Sequence[Union[Num, np.ndarray]], np.ndarray]) -> int:
    return int(np.argmax(seq))


def monotonic(seq: Union[Sequence[Num], np.ndarray], increasing: bool = True):
    arr = np.array(seq)
    diff = arr[1:] - arr[:-1]
    if increasing:
        return np.all(diff > 0)
    else:
        return np.all(diff < 0)


T = TypeVar("T")


def most_common(vals: Iterable[T]) -> Optional[T]:
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def mode(arr: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, int]:
    if arr.size == 1:
        return arr[0], 1
    elif arr.size == 0:
        raise ValueError("Attempted to find mode on an empty array")
    try:
        axis = [i for i in range(arr.ndim)][axis]
    except IndexError:
        raise ValueError("Axis %i out of range for array with %i dimension(s)" % (axis, arr.ndim))
    srt = np.sort(arr, axis=axis)
    dif = np.diff(srt, axis=axis)
    shape = [i for i in dif.shape]
    shape[axis] += 2
    indices = np.indices(shape)[axis]
    index = tuple([slice(None) if i != axis else slice(1, -1) for i in range(dif.ndim)])
    indices[index][dif == 0] = 0
    indices.sort(axis=axis)
    bins = np.diff(indices, axis=axis)
    location = np.argmax(bins, axis=axis)
    mesh = np.indices(bins.shape)
    index = tuple([slice(None) if i != axis else 0 for i in range(dif.ndim)])
    index = [mesh[i][index].ravel() if i != axis else location.ravel() for i in range(bins.ndim)]
    counts = bins[tuple(index)].reshape(location.shape)
    index[axis] = indices[tuple(index)]
    modals = srt[tuple(index)].reshape(location.shape)
    return modals, counts


def modefilt(seq: Union[Sequence[Num], np.ndarray], filtersize: int) -> np.ndarray:
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    if filtersize < 3 or filtersize % 2 == 0:
        raise ValueError(f"Mode filter length ({filtersize}) must be odd")
    if seq.ndim != 1:
        raise ValueError("Input must be one-dimensional")
    if len(seq) < filtersize:
        return seq

    k = (filtersize - 1) // 2
    y = np.zeros((len(seq), filtersize), dtype=seq.dtype)
    y[:, k] = seq
    for i in range(k):
        j = k - i
        y[j:, i] = seq[:-j]
        y[:j, i] = seq[0]
        y[:-j, -(i + 1)] = seq[j:]
        y[-j:, -(i + 1)] = seq[-1]
    return mode(y, axis=1)[0]


def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)


def closest_index(lst: Sequence[float], target: float) -> int:
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect.bisect_left(lst, target)
    if pos == 0:
        return 0
    elif pos == len(lst):
        return len(lst) - 1
    # before = lst[pos - 1]
    # after = lst[pos]
    # if after - target < target - before:
    #     return pos - 1
    # else:
    #     return pos
    return pos


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    (idx,) = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the final element
        idx = np.r_[idx, condition.size - 1]

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx
