from typing import Callable, Tuple

import numpy as np


def symmetric_window(value: float, tolerance: float) -> Tuple[float, float]:
    """Return the symmetric absolute window ``(value - tolerance, value + tolerance)``."""
    return value - tolerance, value + tolerance


class SortedValueIndex:
    """A sorted numeric index supporting fast tolerance-window lookups.

    Sorting the reference values once lets each query binary-search its
    tolerance window in O(log N + k) instead of scanning all N values, which
    pays off when many queries are matched against the same reference set.

    The window shape is delegated to ``bounds_func``, so the same index serves
    m/z matching (a parts-per-million window via :func:`~tools.utils.get_ppm_range`)
    and retention-time matching (a symmetric absolute window) alike.
    """

    def __init__(
        self,
        values: np.ndarray,
        bounds_func: Callable[[float, float], Tuple[float, float]] = symmetric_window,
    ):
        """
        Build the index from a set of reference values.

        Parameters
        ----------
        values : np.ndarray
            Reference values (e.g. m/z or retention times). NaNs sort to the
            end and so never fall inside a finite window.
        bounds_func : Callable
            A function that maps a query to its tolerance window. Called as
            ``bounds_func(value, tolerance)`` and must return the
            ``(lower_bound, upper_bound)`` of the (inclusive) window. Defaults
            to a symmetric absolute window.
        """
        self._bounds_func = bounds_func
        values = np.asarray(values, dtype=float)
        self._order = np.argsort(values)
        self._sorted_values = values[self._order]

    def search(self, value: float, tolerance: float) -> np.ndarray:
        """
        Return the positions of reference values within ``tolerance`` of ``value``.

        Parameters
        ----------
        value : float
            Query value.
        tolerance : float
            Tolerance defining the (inclusive) window; its meaning is set by
            ``bounds_func`` (e.g. parts-per-million for a ppm window, or the
            absolute half-width for a symmetric window).

        Returns
        -------
        np.ndarray
            Integer positions into the original (unsorted) reference array,
            in ascending order.
        """
        lower_bound, upper_bound = self._bounds_func(value, tolerance)
        lo = np.searchsorted(self._sorted_values, lower_bound, side="left")
        hi = np.searchsorted(self._sorted_values, upper_bound, side="right")
        return np.sort(self._order[lo:hi])
