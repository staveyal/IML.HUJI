import sys

sys.path.append("/Users/staveyal/code/university/IML.HUJI")

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian

# from .IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

SAMPLE_SIZE = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    samples = np.random.normal(mu, sigma, SAMPLE_SIZE)

    gaussian = UnivariateGaussian(False)
    gaussian.fit(samples)
    print(f"({gaussian.mu_}, {gaussian.var_})")

    # Question 2 - Empirically showing sample mean is consistent

    x = list(range(10, 1010, 10))
    y = []
    for sample_size in x:
        gaussian.fit(samples[:sample_size])
        y.append(abs(gaussian.mu_ - 10))

    fig = go.Figure(
        data=[go.Scatter(x=x, y=y, mode="markers+lines")],
        layout=go.Layout(
            title=r"(2) Distance between mean and expectation as a function of samples",
            yaxis_title="$|\hat{\mu} - \mu|$",
            xaxis_title="$m\\text{ - number of samples}$",
        ),
    )

    fig.show()
    # with open("q2.jpg", "wb") as f:
    #     f.write(pio.to_image(fig, format="jpg", scale=2.5))

    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure(
        data=[go.Scatter(x=samples, y=gaussian.pdf(samples), mode="markers")],
        layout=go.Layout(
            title=r"(3) Fit PDF value as a function of the sample",
            yaxis_title=r"$f_{\mathbf{\theta}}(x)$",
            xaxis_title=r"$x$ - sample value",
        ),
    ).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    true_cov = np.array(
        [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    )
    samples = np.random.multivariate_normal(
        np.array([0, 0, 4, 0]),
        true_cov,
        SAMPLE_SIZE,
    )
    gaussian = MultivariateGaussian()
    gaussian.fit(samples)

    print(f"mu = {gaussian.mu_}")
    print("cov:")
    for line in gaussian.cov_:
        print(line)

    f1_max = 0
    f3_max = 0
    space = list(np.linspace(-10, 10, 200))
    # Question 5 - Likelihood evaluation
    log_values = [
        [
            MultivariateGaussian.log_likelihood([f1, 0, f3, 0], true_cov, samples)
            for f3 in space
        ]
        for f1 in space
    ]

    current_max = log_values[0][0]
    for i, line in enumerate(log_values):
        for j, item in enumerate(line):
            if item > current_max:
                f1_max = space[i]
                f3_max = space[j]
                current_max = item

    go.Figure(
        data=go.Heatmap(
            x=space,
            y=space,
            z=log_values,
        ),
        layout=go.Layout(
            title=r"(5) Heatmap of log-likelihood as a function of the expectation",
            xaxis_title="$f_3$",
            yaxis_title="$f_1$",
        ),
    ).show()  # x=np.linspace(-10, 10, 200), y=np.linspace(-10, 10, 200))

    # Question 6 - Maximum likelihood
    print(f"(6) Maximum likelihood: ({f1_max}, 0, {f3_max}, 0)")


if __name__ == "__main__":
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
