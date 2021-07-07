# ========================================
# Setup & Design
# ========================================
import os
import json
import pickle
import uuid
import re
import time
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader.yahoo.options import Options
from yahoo_fin import stock_info as si
from yahoo_fin.stock_info import *
from yahoo_fin.stock_info import get_quote_table
import yahoo_fin.stock_info as si
import yfinance as yf
import pyfolio as pf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st 
from prophet import Prophet
import prophet.plot as fplt
from prophet.plot import plot_plotly, plot_components_plotly
import datetime
import datetime as dt
from datetime import datetime
from datetime import datetime, timedelta 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from PIL import  Image
import streamlit as st
# ========================================   
# Data Funktions
# ========================================   
class ticker_data:
    def __init__(self, ticker, start=None, end=None):
        time.sleep(0.3) 
        self.ticker = ticker
        try:
            self._ticker = yf.Ticker(self.ticker)
            if not (start or end):
                self.df = self.df_ = self._ticker.history(period='max', auto_adjust=True)
            else:
                self.df = self.df_ = self._ticker.history(start=start, end=end, auto_adjust=True)
        except Exception as err:
            print(err)

def get_ticker_data():
    tik = ticker_data(snp_ticker)
    return tik
# ========================================
# Prophet
# ========================================
def prophet_df(stk_price):
    df=stk_price.reset_index()
    df = df[["Date","Close"]] 
    df = df.rename(columns = {"Date":"ds","Close":"y"}) 
    return df

def predict_with_prophet():
    stk = get_data()
    stk_df = stk.df["2010":]
    df = prophet_df(stk_df)
    return df

def load_snp_stocks():
    snp=tickers_sp500()
    prices = yf.download(tickers, start='2019-01-01')['Adj Close']
    return prices
# ========================================
# SP 500 App
# =======================================
def app():
    st.markdown('## Probabilistic Programming in Finance')
    st.markdown('### 1. Risk Management with Kelly Criterion')     
    st.markdown('There are two main parts to the Kelly Criterion (KC): win probability and win/loss ratio. In math formula form, that would be:')
    st.markdown('$K_{pct} = W - (1-W)/ R$') 
    st.markdown('**where**')
    st.markdown('$K_{pct}$ = The Kelly percentage')
    st.markdown('$W$ = Winning probability')
    st.markdown('$R$ = Win/loss ratio')
    st.markdown('The KC has more applications than just gambling. It can apply to any decision where we know the values of the variables.')
    st.markdown('#### 1.1. Rigged Coin Flip')
    st.markdown('Enough background talk, let’s get to some coding examples! Now, we’ll go over how to apply KC, when you bet with a coin flip that is favored to you by 55% (a fair coin would have a 50% win chance). For our example, let’s assume you win $1 or lose $1 for the heads tails outcome that is related to how much risked. In other words, we are assuming 1 to 1 payoff relative to our KC or risked amount per bet.')
    p = 0.55
    st.markdown("p = 0.55 fixes the probability for heads.")
    f = p - (1-p)
    st.markdown("f = p - (1-p)")
    st.write(f)
    st.markdown("The above is optimal Kelly Criterion bet size (f). This means that for a 1 to 1 payoff and a 55% favored chance to win, we should risk 10% of our total capital for maximizing our profit.")
    st.markdown('#### 1.2. Simulation of Coin Flips with Variables')
    st.markdown("Number of series to be simulated: 50")
    st.markdown("Number of trials per series: 100")
    def run_simulation(f):
        c = np.zeros((n, I)) #Instantiates an ndarray object to store the simulation results.
        c[0] = 100 #Initializes the starting capital with 100.
        for i in range(I): #Outer loop for the series simulations.
            for t in range(1,n): #Inner loop for the series itself.
                o = np.random.binomial(1, p) #Simulates the tossing of a coin.
                if o > 0: #If 1, i.e., heads …
                    c[t, i] = (1+f) * c[t-1,i] #… then add the win to the capital.
                else: #If 0, i.e., tails …
                    c[t, i] = (1-f) * c[t-1,i] #… then subtract the loss from the capital.
        return c
    # Preparing our simulation of coin flips with variables
    I = 50 #The number of series to be simulated.
    n = 100 #The number of trials per series.
    c_1 = run_simulation(f)
    st.write(c_1.round(2))

    
