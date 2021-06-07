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
from yahoo_fin.stock_info import get_quote_table
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
import streamlit_theme as stt
stt.set_theme({'primary': '#E694FF'})    
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
            
def get_ticker_data():
    tik = ticker_data(ticker_input)
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
# ========================================
# Ticker App
# ========================================
def main():

    st.title(":chart_with_upwards_trend: Ticker Data")
    st.text('Analyse Ticker Data')
    
    ticker_input = st.text_input('Ticker')
    
    @st.cache(persist=True)
    def load_data():
        stk = ticker_data(ticker_input)
        stk_history = stk.df.tz_localize('utc')
        stk_returns = stk_history.Close.pct_change().dropna()
        return  stk_history, stk_returns
    
    df_stk, df_pct = load_data()
    
    df_perf_stats = pf.timeseries.perf_stats(df_pct)
    st.write(df_perf_stats)
    df = df_perf_stats
    
    if st.checkbox("Buy & Hold Return"):
        year = st.date_input("Buy-In Date (YYYY-MM-D)") 
        stock_df = ticker_data(ticker_input)
        stock_df = stock.df[year:]
        stock_df ['LogRets'] = np.log(stock_df['Close'] / stock_df['Close'].shift(1))
        stock_df['Buy&Hold_Log_Ret'] = stock_df['LogRets'].cumsum()
        stock_df['Buy&Hold_Return'] = np.exp(stock_df['Buy&Hold_Log_Ret'])
        font_1 = {'family' : 'Arial', 'size' : 12}
        fig2 = plt.figure()
        plt.style.use('dark_background')
        plt.title(SNP_ticker + ' Buy & Hold', fontdict = font_1)
        plt.plot(stock_df[['Buy&Hold_Return']])
        st.pyplot(fig2)
        st.dataframe(stock_df[['Buy&Hold_Rendite']])        
    
    fig = pf.tears.create_interesting_times_tear_sheet(df_pct, return_fig=True)
    st.write(fig)