## NIFTY50 Portfolio Optimization

This project aims at optimizing the portfolio consisting of NIFTY 50 companies using Deep Learning based **Long-Short-Term-Memory**(LSTM) Network coupled with Univariate Garch Model and Markowitz Model.

#### About Markowitz Model
Markowitz model, also known as mean-variance model was introduced by Harry markowitz in 1952, which is a portfolio optimztion model. It aims to create the most return-to-risk efficient portfolio by analyzing various portfolio combinations based on expected returns (mean) and standard deviations (variance) of the assets.

##### Markowitz Assumptions:
*  The risk of the portfolio is based on its volatility (and covariance) of returns.
*  Analysis is based on a single-period model of investment.
*  An investor is rational, averse to risk and prefers to increase consumption.
*  An investor either minimizes their risk for a given return or maximizes their portfolio return for a given level of risk.

Finding the most optimized portfolio is a two step process begining with finding the set of efficient portfolios followed by picking one from the set with maximum return for a given risk or lowest risk for given return.

* Efficient Frontier
a hyperbola representing portfolios with all the different combinations of assets that result into efficient portfolios (i.e. with the lowest risk, given the same return and portfolios with the highest return, given the same risk). Risk is depicted on the X-axis and return is depicted on the Y-axis. The area inside the efficient frontier (but not directly on the frontier) represents either individual assets or all of their non-optimal combinations.

Now to compute the Efficeint Frontier we have used pypotfolioopt linrary in python. pyportfolioopt simplifies the implementation of the Markowitz Mean-Variance Model to optimize portfolios. It allows investors to find the optimal allocation weights according to many goals and risk tolerance. In this case, we are going to optimize a portfolio to obtain the highest Sharpe ratio possible. 

There are two requisites for this optimization - 
* Expected Returns of all individual assets
* Risk model

In our case, we have used LSTM model for forecasting expected returns. For risk model, one of the most widely used one is covariance matrix which describes the volatility of the assets and the degree to which they are co-dependent.

Let's discuss the two inputs one by one in detail, but before starting, lets look into the dataset and training/validation/testing periods

##### Dataset


##### Asset Covairance Forecast

The covariance matrix is made up of two components - Rolling Volatility & Rolling Correlations. For this project, we have kept the correlations constant based on our training data. Now predicting volatility is bit complex as the volatility exhibits heteroskedasticity meaning the variance is not constant over time. Moreover volatility clustering is often observed meaning periods of low volatility tend to be followed by periods of low volatility and periods of high volatility tend to be followed by periods of high volatility.

In view of above reasons, we trained a GARCH (General Autoregressive Conditional Heteroskedasticity) model that takes into account the volatility clustering. GARCH models the volatility based on past variances (autoregressive: p) and past residual errors (moving average: q). We have found optimal parameters p & q for each asset based on Mean Squared Error over the validation period (2022-07-01 to 2022-12-31)


































