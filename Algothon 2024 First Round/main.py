import pandas as pd
import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    
    price_data = pd.DataFrame(prcSoFar)

    #Price Rate of Change
    p = price_data.loc[:,'0':'49']
    n = price_data.loc[:,'0':'49'].shift(periods=10)
    Instroc = ((p-n)/n) * 100
    roc = Instroc.dropna()

    roc3 = roc.iloc[:,0:50].shift(periods=-10)
    rocn = roc.iloc[:,0:50]

    #Relative Strength Index
    r = price_data.loc[:,'0':'49']
    delta = r.diff(1)
    delta.dropna(inplace=True)

    positive = delta.copy()
    negative = delta.copy()

    positive[positive>0]=0
    negative[negative<0]=0

    days = 10

    average_gain = positive.rolling(window=days).mean()
    average_loss = abs(negative.rolling(window=days).mean())

    relative_strength = average_gain/average_loss
    RSI = 100.0 - (100.0/(1.0+relative_strength))

    alpha = 2 / (days + 1)  # EMA smoothing factor
    smoothed_RSI = RSI.copy()
    smoothed_RSI.iloc[days:] = RSI.iloc[days:].ewm(alpha=alpha, adjust=False).mean()

    #Ema 5 and Ema 10 CrossOver
    ema5 = price_data.iloc[:,0:50].ewm(span=5, adjust=False).mean()
    ema10 = price_data.iloc[:,0:50].ewm(span=10, adjust=False).mean()

    #Risk Management
    every_day_trade_limit = 0.20

    for row in range(roc3.shape[0]):
        for col in range(roc3.shape[1]):
            if smoothed_RSI.iloc[row, col] >= 70 and ema5.iloc[row, col] <= ema10.iloc[row, col]:
                if roc3.iloc[row, col] < rocn.iloc[row, col] and rocn.iloc[row, col] >= 0:
                    currentPos[col] = -10000 * every_day_trade_limit
            if smoothed_RSI.iloc[row, col] <= 30 and ema5.iloc[row, col] >= ema10.iloc[row, col]:
                if roc3.iloc[row, col] > rocn.iloc[row, col] and rocn.iloc[row, col] <= 0:
                    currentPos[col] = 10000 * every_day_trade_limit
            else: 
                currentPos[col] == 0
    return currentPos