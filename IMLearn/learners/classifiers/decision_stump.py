from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> None:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # Use _find_threshold to find the best feature and threshold by which to split
        # the data according to the CART algorithm
        # Set self.threshold_, self.j_ and self.sign_ according to the results
        # Note: self.sign_ should be either 1 or -1
        # Note: self.j_ should be an integer between 0 and n_features - 1
        # Note: self.threshold_ should be a float

        # self.fitted_ = True
        error = np.finfo(np.float64).max
        for i, sign in product(range(X.shape[1]), [-1, 1]):
            thresh, curr_thresh_error = self._find_threshold(X[:, i], y, sign)

            if curr_thresh_error < error:
                self.threshold_, self.j_, self.sign_, error = thresh, i, sign, curr_thresh_error

        self.fitted_ = True



    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        distances = np.sign(X[:, self.j_] - self.threshold_)
        distances[distances == 0] = 1
        return  distances * self.sign_

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # Sort both values and labels according to values
        indices = np.argsort(values)
        values, labels = values[indices], labels[indices]

        minus_inf_loss = np.sum(np.abs(labels)[np.sign(labels) == sign])

        losses = np.append(minus_inf_loss, np.cumsum(np.abs(labels)[np.sign(labels) == -sign]))

        # Find the threshold minimizing the loss
        thr_index = np.argmin(losses)
        return np.concatenate([[-np.inf], values[1:], [np.inf]])[thr_index], losses[thr_index]
        


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
