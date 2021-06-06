# ========================================
# Setup and Design
# ========================================
import streamlit as st 
import time
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader.yahoo.options import Options
from yahoo_fin import stock_info as si
import yfinance as yf
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
from yahoo_fin.stock_info import get_quote_table
import warnings
import pyfolio as pf
# ========================================     
# Data Funktions
# ========================================   
class ticker_data:
    def __init__(self, ticker, start=None, end=None):
        time.sleep(6) 
        self.ticker = ticker
        try:
            self._ticker = yf.Ticker(self.ticker)
            if not (start or end):
                self.df = self.df_ = self._ticker.history(period='max', auto_adjust=True)
            else:
                self.df = self.df_ = self._ticker.history(start=start, end=end, auto_adjust=True)
        except Exception as err:
            print(err)
# ========================================   
def py_data():
    ticker = get_data().df
    ticker.index = ticker.index.tz_localize('utc')
    return ticker

def get_data():
    stk = ticker_data(ticker_input)
    return stk

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
# ========================================
# Launche App
# ========================================
# Create an instance of the app 
st.title(":chart_with_upwards_trend: Data Application")
st.subheader("Ticker Data")  
st.markdown("### enter a ticker to start analysis.") 
ticker_input = st.text_input('Ticker')
data_ = py_data()
returns = data_.Close.pct_change().dropna()
heatmap = pf.plotting.plot_monthly_returns_heatmap(returns)
st.pyplot(heatmap)
times=pf.tears.create_interesting_times_tear_sheet(returns, return_fig=True)
st.write(times)
