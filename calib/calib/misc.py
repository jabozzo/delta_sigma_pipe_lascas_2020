#! /usr/bin/env python

import copy
import json

import numpy as np


NO_PARAM = object()
"""
Object to use instead of None when None is a valid value and cannot be used as
default.
"""


def default(value, default):
    """
    Same as default if value is None else value, but less efficient.

    :param value: Value to return if not None.
    :param default: Value to return if value is None.
    """
    return default if value is None else value


def push_random_state(state=None):
    """
    Returns a context manager that will reset the numpy.random state once the
    context is left to the state that was before entering the context. Useful
    for setting a seed or another state temporarily.

    :param state: A valid numpy random state

        .. seealso :: :func:`numpy.random.get_state`
    :type state: tuple
    """
    class StateStore(object):
        def __init__(self, state):
            self.state = copy.deepcopy(state)

        def __enter__(self):
            self._prev_state = np.random.get_state()
            if self.state is not None:
                np.random.set_state(self.state)
            return self

        def __exit__(self, type, value, tb):
            self.state = copy.deepcopy(np.random.get_state())
            np.random.set_state(self._prev_state)

    return StateStore(state)


class Namespace(object):
    def __init__(self, **kwargs):
        self._my_attrs = set()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, name, value):
        if name != "_my_attrs":
            self._my_attrs.add(name)

        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name != "_my_attrs":
            self._my_attrs.discard(name)

        super().__delattr__(name)

    def __copy__(self, memo=None):
        return Namespace(**{name : getattr(self, name) for name in self._my_attrs})

    def __deepcopy__(self, memo=None):
        return Namespace(**{name : copy.deepcopy(getattr(self, name), memo)
            for name in self._my_attrs})


def getitem(iterable, idx):
    if len(idx) > 1:
        return getitem(iterable[idx[0]], idx[1:])
    elif len(idx) == 1:
        return iterable[idx[0]]
    else:
        return iterable


def setitem(iterable, idx, value):
    if len(idx) > 1:
        setitem(iterable[idx[0]], idx[1:], value)
    elif len(idx) == 1:
        iterable[idx[0]] = value
    else:
        raise ValueError("Does not work on scalars")


def multiget(dct, *keys):
    return tuple(dct[key] for key in keys)


def json_args(lst):
    result = []

    for element in lst:
        element = json.loads(element)
        if isinstance(element, str): # If quotes where escaped
            element = json.loads(element)
        assert isinstance(element, dict)
        result.append(element)

    return result


def ogrid(idx, length, tot_len=None):
    tot_len = default(tot_len, idx + 1)
    assert tot_len >= idx + 1
    shape = (1,)*idx + (length,) + (1,)*(tot_len - idx - 1)
    return np.reshape(list(range(length)), shape)


def iterate_combinations(n, k):
    if k > n:
        raise ValueError("k ({}) must be less or equal than n ({}).".format(k, n))
    elif k <= 0:
        raise ValueError("k ({}) must be greater than 0.".format(k))
    state = tuple(range(k))

    def update(state, index):
        inv_index = k-index
        limit = n - inv_index
        if state[index] >= limit:
            return update(state, index-1)
        else:
            start_point = state[index] + 1
            return state[0:index] + tuple(range(start_point, start_point + inv_index))

    while state[0] <= n - k:
        yield state
        try:
            state = update(state, k-1)
        except IndexError:
            break


def iterate_permutations(lst):
    if len(lst) > 1:
        for ii in range(len(lst)):
            element = lst[ii:ii+1]
            rest = lst[:ii] + lst[ii+1:]
            for combination in iterate_permutations(rest):
                yield element + combination

    elif len(lst) == 1:
        yield lst[0:1]

    else:
        yield []
