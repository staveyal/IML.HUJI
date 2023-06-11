from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
import tqdmthe following problem is: lim $\sin$

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    X = X.drop(X[X["zipcode"] == 0].index)
    X = pd.get_dummies(X, columns=["zipcode"], prefix="zipcode")
    X["sqft_living_diff"] = X["sqft_living"] - X["sqft_living15"]
    X["sqft_living_lot_diff"] = X["sqft_lot"] - X["sqft_lot15"]
    X = X.drop(columns=["id", "date", "lat", "long", "sqft_lot15", "sqft_living15"])

    X = X.dropna()

    if y is None:
        if "price" in X:
            X = X.drop(X[X["price"] <= 0].index)
        return X
    
    # We drop the rows with missing values, and price = 0 with the corresponding rows in y
    y = y.loc[X.index]
    y = y.drop(y[y <= 0].index)
    y = y.dropna()
    return X.loc[y.index], y



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col in tqdm.tqdm(X.columns):
        fig = px.scatter(x=X[col], y=y, trendline="ols")
        # corr = np.cov(X[col], y)/(np.std(X[col]) * np.std(y))
        # Manually calculate pearson correlation
        corr = y.cov(X[col])/(X[col].std() * y.std())
        corr = round(corr, 2)
        fig.update_layout(title=f"{col} vs. Price (Pearson Correlation: {str(corr)})",
                          xaxis_title=col,
                          yaxis_title="Price")
        fig.write_image(f"{output_path}/{col}.png")
    


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")
    # df = pd.read_csv("datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_X, train_Y, test_X, test_Y = split_train_test(df.drop(columns=["price"]), df["price"])

    # Question 2 - Preprocessing of housing prices dataset
    # df = preprocess_data(df)
    train_X, train_Y = preprocess_data(train_X, train_Y)
    test_X, test_Y = preprocess_data(test_X, test_Y)

    # Question 3 - Feature evaluation with respect to response
    # feature_evaluation(train_X, train_Y, output_path=".")

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    model = LinearRegression()    
    mean_losses, var_losses = [], []
    for percentage in tqdm.tqdm(np.arange(10, 101, 1)):
        losses = []
        for _ in np.arange(1,10,1):
            x = train_X.sample(frac=percentage/100)
            model.fit(x, train_Y[x.index])
            loss = model.loss(test_X, test_Y)
            losses.append(loss)
        mean = np.mean(losses)
        mean_losses.append(mean)
        var_losses.append(np.std(losses, ddof=1))

    mean_losses, var_losses = np.array(mean_losses), np.array(var_losses)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arange(10, 101, 1),
                             showlegend=False,
                             y=mean_losses-2*var_losses, mode="lines",
                             line={"color": "lightgray"},
                             name="Mean Loss - 2*Std"))
    fig.add_trace(go.Scatter(x=np.arange(10, 101, 1),
                             showlegend=False,
                             line={"color": "lightgray"},
                             fill="tonexty",
                             y=mean_losses+2*var_losses, mode="lines",
                             name="Mean Loss + 2*Std"))
    fig.add_trace(go.Scatter(x=np.arange(10, 101, 1), y=mean_losses,
                             marker={"color": "black", "size": 1},
                             showlegend=False,
                             mode="markers+lines", name="Mean Loss"))

    # add titles for the axis
    fig.update_layout(xaxis_title="Training Size (%)",
                        yaxis_title="Mean Loss",
                        title="MSE as Function of Training Size")

    fig.show("browser")

# go.Figure([go.Scatter(x=ms, y=means-2*variances, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
#            go.Scatter(x=ms, y=means+2*variances, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False),
#            go.Scatter(x=ms, y=means, mode="markers+lines", marker=dict(color="black",size=1), showlegend=False)],
    

            

        
    


