from typing import Callable

import numpy as np


class SortedValueIndex:
    """A sorted m/z index supporting fast ppm-tolerance window lookups.

    Sorting the reference m/z values once lets each query binary-search its ppm
    window in O(log N + k) instead of scanning all N values, which pays off when
    many query m/z values are matched against the same reference set.
    """

    def __init__(self, mz: np.ndarray, bounds_func: Callable, abs_tol: float = 0):
        """
        Build the index from a set of reference m/z values.

        Parameters
        ----------
        mz : np.ndarray
            Reference m/z values. NaNs sort to the end and so never fall
            inside a finite ppm window.
        bounds_func : Callable
            A function that computes the tolerance window for a query value.
            Called as ``bounds_func(mz, ppm_error, abs_tol)`` and must return
            the ``(lower_bound, upper_bound)`` of the (inclusive) window.
        abs_tol : float
            Absolute tolerance added to every window, on top of the ppm term.
        """

        self._bound_func = bounds_func
        self._abs_tol = abs_tol
        mz = np.asarray(mz, dtype=float)
        self._order = np.argsort(mz)
        self._sorted_mz = mz[self._order]

    def search(self, mz: float, ppm_error: float) -> np.ndarray:
        """
        Return the positions of reference values within ``ppm_error`` of ``mz``.

        Parameters
        ----------
        mz : float
            Query m/z value.
        ppm_error : float
            Mass tolerance in parts-per-million defining the (inclusive) window.

        Returns
        -------
        np.ndarray
            Integer positions into the orifunginal (unsorted) reference array,
            in ascending order.
        """
        lower_bound, upper_bound = self._bound_func(mz, ppm_error, self._abs_tol)
        lo = np.searchsorted(self._sorted_mz, lower_bound, side="left")
        hi = np.searchsorted(self._sorted_mz, upper_bound, side="right")
        return np.sort(self._order[lo:hi])
