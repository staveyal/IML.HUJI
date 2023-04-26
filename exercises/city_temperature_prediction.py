import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

from functools import partial
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df.dropna()
    df["DayOfYear"] = df["Date"].dt.day_of_year
    df = df.drop(df[df["Temp"] < -40].index)

    return df



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")
    #
    # Question 2 - Exploring data for specific country
    israel_df = df.loc[df["Country"] == "Israel"]
    israel_df["Year"] = israel_df["Date"].dt.year.astype(str)
    fig = px.scatter(israel_df, x="DayOfYear", y="Temp",
                     color="Year")
    fig.update_layout(title="Temp. in Israel depending on day of year")
    fig.show("browser")

    israel_df["Month"] = israel_df["Date"].dt.month
    std_unbiased = partial(np.std, ddof=1)
    month_df = israel_df.groupby("Month").Temp.agg(std_unbiased)
    fig = px.bar(x=month_df.index, y=month_df.array)
    fig.update_layout(title="Standard deviation of temp. in Israel by Month",
                      xaxis_title="Month",
                      yaxis_title="Deviation")
    fig.show("browser")

    # Question 3 - Exploring differences between countries
    df["Month"] = df["Date"].dt.month
    group_by_country = df.groupby(["Month", "Country"], group_keys=False).Temp.agg([np.mean, std_unbiased]).reset_index()
    # group_by_month = df.groupby("Month").Temp.agg({"mean": np.mean, "std": std_unbiased})
    # fig = px.line(x=group_by_country.index.levels[1], y="mean", color=group_by_country.index.levels[0])#, error_y=)
    fig = px.line(group_by_country, x="Month", y="mean", color="Country", error_y="std")
    fig.update_layout(yaxis_title="Mean temp.",
                      title="Mean temp. by month of different countries")
    fig.show("browser")


    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_df.drop(columns=["Temp"]), df["Temp"])
    losses = []
    for k in np.arange(1, 11, 1):
        polynomial = PolynomialFitting(k)
        polynomial.fit(train_X["DayOfYear"], train_y)
        losses.append(round(polynomial.loss(test_X["DayOfYear"], test_y), 2))

    print(losses)
    fig = px.bar(x=np.arange(1, 11, 1), y=losses)
    fig.update_layout(title="Loss by the degree of the polynomial fitting k",
                      xaxis_title="k",
                      yaxis_title="MSE")
    fig.show("browser")

    # Question 5 - Evaluating fitted model on different countries
    k = 5
    israel_fit = PolynomialFitting(k)
    israel_fit.fit(israel_df["DayOfYear"], israel_df["Temp"])

    other_countries = df.drop(df[df["Country"] == "Israel"].index)
    country_losses = {
        country: israel_fit.loss(df[df["Country"] == country]["DayOfYear"], df[df["Country"] == country]["Temp"])
        for country in set(other_countries["Country"])
    }

    fig = px.bar(x=country_losses.keys(), y=country_losses.values())
    fig.update_layout(title="Loss by country",
                        xaxis_title="Country",
                        yaxis_title="MSE")
    fig.show("browser")