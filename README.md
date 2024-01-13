## NIFTY50 Portfolio Optimization

This project aims at optimizing the portfolio consisting of NIFTY 50 companies using Deep Learning based **Long-Short-Term-Memory**(LSTM) Network coupled with Univariate Garch Model and Markowitz Model.

### About Markowitz Model
Markowitz model, also known as mean-variance model was introduced by Harry markowitz in 1952, which is a portfolio optimztion model. It aims to create the most return-to-risk efficient portfolio by analyzing various portfolio combinations based on expected returns (mean) and standard deviations (variance) of the assets.

#### Markowitz Assumptions:
*  The risk of the portfolio is based on its volatility (and covariance) of returns.
*  Analysis is based on a single-period model of investment.
*  An investor is rational, averse to risk and prefers to increase consumption.
*  An investor either minimizes their risk for a given return or maximizes their portfolio return for a given level of risk.

Finding the most optimized portfolio is a two step process begining with finding the set of efficient portfolios followed by picking one from the set with maximum return for a given risk or lowest risk for given return.

##### Efficient Frontier
a hyperbola representing portfolios with all the different combinations of assets that result into efficient portfolios (i.e. with the lowest risk, given the same return and portfolios with the highest return, given the same risk). Risk is depicted on the X-axis and return is depicted on the Y-axis. The area inside the efficient frontier (but not directly on the frontier) represents either individual assets or all of their non-optimal combinations.




<figure>
    <p align="center">
       <img src="https://github.com/himanshu1311/nifty50_portfolio_optimization/assets/58727871/991d913c-be02-4b6f-850d-7e08d0eda0f0"
             height="80%"/>
    </p>
    <p align="center">
      <em> Efficient Frontier (Source: quantpedia.com) </em>
    </p>
</figure>

Now to compute the Efficeint Frontier we have used pypotfolioopt linrary in python. pyportfolioopt simplifies the implementation of the Markowitz Mean-Variance Model to optimize portfolios. It allows investors to find the optimal allocation weights according to many goals and risk tolerance. In this case, we are going to optimize a portfolio to obtain the highest Sharpe ratio possible. 

There are two requisites for this optimization - 
* Expected Returns of all individual assets
* Risk model

In our case, we have used LSTM model for forecasting expected returns. For risk model, one of the most widely used one is covariance matrix which describes the volatility of the assets and the degree to which they are co-dependent.

Let's discuss the two inputs one by one in detail, but before starting, lets look into the dataset and training/validation/testing periods

### Dataset

We downloaded the NIFTY50 companies data starting Jan-2013. There are 47 companies in NIFTY50 list dating back this long. We used closing price of each trading day to compute daily returns. 


<figure>
    <p align="center">
       <img src="https://github.com/himanshu1311/nifty50_portfolio_optimization/assets/58727871/f8824079-0941-4e83-8e5c-ce6eec7768c9"/>
    </p>
    <p align="center">
      <em> NIFTY 50 Companies daily returns snapshot </em>
    </p>
</figure>

#### Covid Correction

Eary 2020 saw a huge spike in market returns. If we see the total portfolio returns (eqaul weights), there is huge spike starting march till june of 2020 (can be seen in the plot below). We often treat this as noise and should be treated before feeding into our downstream models. Following a simple approach we just calculated the day over day growth rate in retuns an year ago (in 2019) and applied the same starting Feb 2020. The assumption here is the market growth will be simialr to what it was an year ago and would follow the same trend if covid wasn't there. Thoough there are several long term and short term events that decides the market dynamics but we wanted to keep this methodology simple for start and leave it for further improvement in future.

<figure>
    <p align="center">
       <img src="https://github.com/himanshu1311/nifty50_portfolio_optimization/assets/58727871/c3775b13-46ef-48ad-811e-426f8b19ba52"/>
    </p>
    <p align="center">
      <em> Portfolio Returns (Before applying correction) </em>
    </p>
</figure>


<figure>
    <p align="center">
       <img src="https://github.com/himanshu1311/nifty50_portfolio_optimization/assets/58727871/2647eafe-183d-4d26-9778-e24ff1090773"/>
    </p>
    <p align="center">
      <em> Portfolio Returns (After applying correction) </em>
    </p>
</figure>

<br>
Once we have our data prepared, we start with calculation of the mmarkowitz inputs. For this project, we have divided our data into 3 segments as:

* **Training**   : 2013-01-01 to 2022-06-30
* **Validation** : 2022-07-01 to 2022-12-31
* **Testing**    : 2023-01-01 to 2023-12-31

### Asset Covariance Forecast

The covariance matrix is made up of two components - Rolling Volatility & Rolling Correlations. For this project, we have kept the correlations constant based on our training data. Now predicting volatility is bit complex as the volatility exhibits heteroskedasticity meaning the variance is not constant over time. Moreover volatility clustering is often observed meaning periods of low volatility tend to be followed by periods of low volatility and periods of high volatility tend to be followed by periods of high volatility.

In view of above reasons, we trained a GARCH (General Autoregressive Conditional Heteroskedasticity) model that takes into account the volatility clustering. GARCH models the volatility based on past variances (autoregressive: p) and past residual errors (moving average: q). We have found optimal parameters p & q for each asset, based on Mean Squared Error over the validation period (2022-07-01 to 2022-12-31). Here is a snapshot of optimal parameters for few assets:


<figure>
    <p align="center">
       <img src="https://github.com/himanshu1311/nifty50_portfolio_optimization/assets/58727871/d7c6e66e-49e7-464f-ba81-cecf05864bf7"/>
    </p>
    <p align="center">
      <em> optimal GARCH parameters p & q for each asset in the portfolio </em>
    </p>
</figure>

<br></br>
We are rebalancing the portfolio every week, so for a particular rebalancing date, we fit the the GARCH model on the asset returns till the given rebalancing date and forecast volatility for next 5 days(1 week). We then take the average volatility forecast of the five days for each asset. once we have volatility(in the form of diagonal matrix) and correlation matrix, we can calculate the covariance matrix as:
<br></br>

<figure>
    <p align="center">
       <img src="https://github.com/himanshu1311/nifty50_portfolio_optimization/assets/58727871/c613c7fe-dcb1-4547-b5a1-058a045c4a3b"/>
    </p>
    <p align="center">
      <em> Covariance matrix calculation </em>
    </p>
</figure>


### Asset Returns Forecast






























