"""
Tools without a common theme.
"""
from copy import copy
from functools import partial
from itertools import chain, cycle, islice, product
from textwrap import indent
indent2 = partial(indent, prefix="  ")

from .typing import Iterable

try:
    get_ipython()
except NameError:
    RUNNING_IN_IPYTHON = False
else:
    RUNNING_IN_IPYTHON = True


def flatten_list(x: Iterable[Iterable[object]]) -> list[object]:
    """
    Flatten a list of lists (or other iterables), using itertools.chain.
    """
    return list(chain.from_iterable(x))


def roundrobin(*iterables):
    """
    Visit input iterables in a cycle until each is exhausted.
    From the itertools recipes.
    """
    iterators = map(iter, iterables)
    for num_active in range(len(iterables), 0, -1):
        iterators = cycle(islice(iterators, num_active))
        yield from map(next, iterators)


def make_iterable(x: object, exclude: Iterable[type]=[str]) -> Iterable:
    """
    Check if x is an iterable.
    If it is an iterable, return it as-is.
    If not, make it into a single-element list and return that.

    Classes listed in `exclude` are not counted as iterable here.
    For example, if exclude=[str], then the string "abc" would count as non-iterable and would be wrapped into a list: ["abc"].
    """
    if isinstance(x, Iterable) and not isinstance(x, *exclude):
        return x
    else:
        return [x]


def dict_product(d: dict) -> list[dict]:
    """
    For a dict `d` with iterable values, return the Cartesian product, i.e. all combinations of values in those iterables.
    The result is a list of dictionaries.
    This is useful for iterating over multiple **kwargs in another function.
    Based on https://stackoverflow.com/a/40623158/2229219
    """
    d_all_iterable = {key: make_iterable(value) for key, value in d.items()}
    d_product = [dict(zip(d_all_iterable.keys(), i)) for i in product(*d_all_iterable.values())]
    return d_product


def invert_dict(d: dict) -> dict:
    """
    Return the inverse of a dictionary, i.e. go from {key: value} to {value: key}.
    Does not check for uniqueness in values, so use with care.
    """
    return {value: key for key, value in d.items()}
