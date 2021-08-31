# Overview
# Dependencies
- Python==3.7
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- gym
- stable-baselines3
- torch==1.9
- yfinance
- ta
- backtesting
- python-dotenv


# Environment

## Input Data Format
The format of the data is Pandas.DataFrame
- Columns: Open, High, Low, Close, Volume
- Index: Date

## Observations
The following features are extracted from the input data

- Trend Indicator
    - MACD (Moving Average Convergence Divergence)
    - DMI (Directional Movement Index)

- Oscillator Indicator
    - RSI (Relative Strength Index)

- Volume Indicator
    - MFI (Money Flow Index)
    - CMF (Chaikin Money Flow)

Observations include these features for the window_size.  
Therefore, the shape of observations is (window_size, features_size).

## Actions
- Buy
    - If the agent doesn't have a position, buy at the closing price.
    - If the agent has a long position, takes no action.
    - If the agent has a short position, close the position at the closing price.
- Sell
    - If the agent doesn't have a position, sell at the closing price.
    - If the agent has a long position, close the position at the closing price.
    - If the agent has a short position, takes no action.
- Neutral (maybe no need)
    - Do nothing

See [here](https://en.wikipedia.org/wiki/Position_(finance)) for the meaning of the position

## Reward Function
![Reward](https://latex.codecogs.com/gif.latex?R=\frac{Entry&space;Price&space;-&space;CurrentPrice}{CurrentPrice})  

# Research Papers
+ [Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review](https://arxiv.org/abs/2106.00123)
+ [Reinforcement learning in financial markets - a survey](https://www.semanticscholar.org/paper/Reinforcement-learning-in-financial-markets-a-Fischer/922864ede84bc49be4ac676951278a9b568b6383#paper-header)
+ [Adaptive Stock Trading Strategies with Deep Reinforcement Learning Methods](https://www.researchgate.net/publication/342155465_Adaptive_Stock_Trading_Strategies_with_Deep_Reinforcement_Learning_Methods)
