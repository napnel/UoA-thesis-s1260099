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

## Actions
- Buy
- Sell
- Hold (maybe no need)

## Reward Function
$$R=\frac{Position Price - CurrentPrice}{CurrentPrice} \times 100$$
See [here](https://en.wikipedia.org/wiki/Position_(finance)) for the meaning of the position

# Trade Rule
ポジションは片方しか持てない。（両建てできない）  
エージェントは、現在を含む過去20本分のOHLCVデータを観測し、現在の価格で売買する。  
取引数量は固定。


# Research Papers
+ [Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review](https://arxiv.org/abs/2106.00123)
+ [Adaptive Stock Trading Strategies with Deep Reinforcement Learning Methods](https://www.researchgate.net/publication/342155465_Adaptive_Stock_Trading_Strategies_with_Deep_Reinforcement_Learning_Methods)
