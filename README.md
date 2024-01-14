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
A hyperbola representing portfolios with all the different combinations of assets that result into efficient portfolios (i.e. with the lowest risk, given the same return and portfolios with the highest return, given the same risk). Risk is depicted on the X-axis and return is depicted on the Y-axis. The area inside the efficient frontier (but not directly on the frontier) represents either individual assets or all of their non-optimal combinations.




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

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture that is particularly effective for sequence prediction tasks, making it well-suited for stock price prediction.

LSTMs are well-suited for stock price prediction due to the temporal nature of financial time series data. Here's how LSTMs are commonly applied in this context:
- Temporal Dependencies: LSTMs can capture and utilize temporal dependencies in stock price movements. This is important because stock prices are often influenced by historical price patterns.
- Feature Extraction: LSTMs can automatically learn relevant features from the input data, which is essential when dealing with complex and high-dimensional financial data.
- Handling Non-linearity: Stock price movements are often nonlinear and influenced by a variety of factors. LSTMs, with their ability to model complex relationships, can handle this non-linearity better than simpler models.
- Sequence-to-Sequence Learning: LSTMs can be trained in a sequence-to-sequence manner, taking historical stock prices as input and predicting future prices as output.
- Handling Varying Time Intervals: LSTMs can adapt to irregularly spaced time intervals in financial data, allowing for more flexibility in modeling.

In order to use LSTM, the input and output data should have a specific shape. In a nutshell, the input in an LSTM model is a three-dimensional array where the first dimension represents the number of samples (or batch size) like the number of rows of data in a two-dimensional setting, the second dimension stands for time steps which indicates the amount of time that we want to go back through time, and the third dimension shows the number of features. So, the input shape looks like [number_of_samples, time_steps, input_dim].

<figure>
    <p align="center">
       <img src="https://github.com/himanshu1311/nifty50_portfolio_optimization/blob/main/Images/LSTM_Shape.png"/>
    </p>
    <p align="center">
      <em> Covariance matrix calculation </em>
    </p>
</figure>

We explore more features which could be helpful in stock price forecasting. We have downloaded two variables from yfinance, Close price and Volume of each stocks. We created new features - Relative Strength Index (RSI), exponential moving averages, Volume-Weighted average price (VWAP) for different period, VWAP change.

-  RSI is a metric which has been found useful while studying movement of stock prices. It is commonly used in technical analysis to identify overbought or oversold conditions in a financial market. The RSI value oscillates between 0 and 100. When RSI values are above a certain threshold (typically 70), it suggests that the stock may be overbought, and there is a higher likelihood of a price correction or reversal. Conversely, when RSI values are below a certain threshold (typically 30), it indicates that the stock may be oversold, and there could be a potential for a price rebound. RSI appears to correlate to Stock Prices, we further categorized RSI values in Low (RSI < 30), Medium( RSI >30, RSI <70) and High category (RSI >70). 
-  VWAP is the ratio of the value of a security or financial asset traded to the total volume of transactions during a trading session.
-  SMA (Simple Moving Averages) and EMA are some well know pointers when it comes to Price Tracking and making decisions based on them. These methods help in identifying trends related to stock prices. While as the name suggests, SMA are just the average of a period where as EMA attach weights to the calculation and sensitive to recent price movements. We have used EMA of the stock price as the input

### Results and Conclusion

























