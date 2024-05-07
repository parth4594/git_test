"""Contains the class for custom cross-validation function to create time
series cross-validation set for pycaret, with added parameter, namely,
min_train_size. The original class from scikit-learn is only modified to
include this particular parameter only.
"""
import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class TimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Read more in the :ref:`User Guide <time_series_split>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 3.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    min_train_size : int, default=None
        Minimum size for a single training set

    max_train_size : int, default=None
        Maximum size for a single training set.

    test_size : int, default=None
        Used to limit the size of the test set. Defaults to
        ``n_samples // (n_splits + 3)``, which is the maximum allowed value
        with ``gap=0``.

        .. versionadded:: 0.24

    gap : int, default=0
        Number of samples to exclude from the end of each train set before
        the test set.

        .. versionadded:: 0.24

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> tscv = TimeSeriesSplit()
    >>> print(tscv)
    TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    >>> for train_index, test_index in tscv.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]
    TRAIN: [0 1 2 3] TEST: [4]
    TRAIN: [0 1 2 3 4] TEST: [5]
    >>> # Fix test_size to 2 with 12 samples
    >>> X = np.random.randn(12, 2)
    >>> y = np.random.randint(0, 2, 12)
    >>> tscv = TimeSeriesSplit(n_splits=3, test_size=2)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 2 3 4 5] TEST: [6 7]
    TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]
    TRAIN: [0 1 2 3 4 5 6 7 8 9] TEST: [10 11]
    >>> # Add in a 2 period gap
    >>> tscv = TimeSeriesSplit(n_splits=3, test_size=2, gap=2)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 2 3] TEST: [6 7]
    TRAIN: [0 1 2 3 4 5] TEST: [8 9]
    TRAIN: [0 1 2 3 4 5 6 7] TEST: [10 11]

    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)
    + n_samples % (n_splits + 1)`` in the ``i`` th split,
    with a test set of size ``n_samples//(n_splits + 1)`` by default,
    where ``n_samples`` is the number of samples.
    """

    def __init__(
        self,
        n_splits=5,
        *,
        min_train_size=None,
        max_train_size=None,
        test_size=None,
        gap=0,
    ):
        """Initilization"""
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap

        test_size = None

        if self.test_size is not None:
            test_size = self.test_size

        if self.min_train_size is not None:
            min_train_size = self.min_train_size
            if test_size is None:
                test_size = (n_samples - gap - min_train_size) // n_splits
        else:
            if test_size is None:
                test_size = n_samples // n_folds
            min_train_size = n_samples - gap - n_splits * test_size

        # Make sure we have enough samples for the given split parameters
        if n_folds > n_samples:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of samples={n_samples}."
            )
        if n_samples - gap - (test_size * n_splits) <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of samples"
                f"={n_samples} with test_size={test_size} and gap={gap}."
            )
        if n_splits < 1:
            raise ValueError(
                f"Too few splits={n_splits} for minimum training size"
                f"{min_train_size} is not less that number of samples \
                    {n_samples}."
            )

        indices = np.arange(n_samples)
        test_starts = range(min_train_size, n_samples, test_size)

        for test_start in test_starts:
            train_end = test_start - gap
            if test_start + test_size > n_samples:
                break
            if self.max_train_size and self.max_train_size < train_end:
                yield (
                    indices[train_end - self.max_train_size : train_end],
                    indices[test_start : test_start + test_size],
                )
            else:
                yield (
                    indices[:train_end],
                    indices[test_start : test_start + test_size],
                )
