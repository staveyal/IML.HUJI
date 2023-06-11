from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        samples, responses = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_callback(classifier: Perceptron, x: np.ndarray, y: int):
            losses.append(classifier.loss(samples, responses))

        perceptron = Perceptron(callback=loss_callback)
        perceptron.fit(samples, responses)

        # Plot figure of loss as function of fitting iteration
        go.Figure(go.Scatter(x=np.arange(1, len(losses) + 1), y=losses, mode="lines")
                  ).update_layout(title=f"Perceptron Model Training Loss - {n} dataset", xaxis_title="Iteration no.", yaxis_title="Loss").show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
    # for f in ["gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("datasets/" + f)

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        print("Done fitting GNB")
        gnb_preds = gnb.predict(X)
        print("Done predicting GNB")

        lda = LDA()
        lda.fit(X, y)
        lda_preds = lda.predict(X)
        print("Done predicting LDA")

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            f"Gaussian Naive Bayes. Accuracy: {accuracy(y, gnb_preds)*100}%",
            f"LDA. Accuracy: {accuracy(y, lda_preds)*100}%"])
        fig.update_layout(title=f"Comparison of Gaussian Classifiers - {f} dataset")
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker_color=y, marker_symbol=gnb_preds, showlegend=False),
                                #  .update_layout(title=f"Accuracy: {accuracy(y, gnb_preds):.2f}"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                      marker_color=y, marker_symbol=lda_preds, showlegend=False),
                                #  .update_layout(title=f"Accuracy: {accuracy(y, lda_preds):.2f}"),
                      row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means



        # # Add ellipses depicting the covariances of the fitted Gaussians
        for c in range(len(np.unique(y))):
            # The marker is a big black x marking the mean of the Gaussian
            fig.add_trace(go.Scatter(x=[gnb.mu_[c][0]], y=[gnb.mu_[c][1]], mode="markers", marker_color="black",
                                     marker_symbol="x", marker_size=17, showlegend=False)
                          , row=1, col=1)
            fig.add_trace(go.Scatter(x=[lda.mu_[c][0]], y=[lda.mu_[c][1]], mode="markers", marker_color="black",
                                     marker_symbol="x", marker_size=17, showlegend=False)
                          , row=1, col=2)
            fig.add_trace(get_ellipse(gnb.mu_[c], np.diag(gnb.vars_[c])), row=1, col=1)
            fig.add_trace(get_ellipse(lda.mu_[c], lda.cov_), row=1, col=2)


        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
