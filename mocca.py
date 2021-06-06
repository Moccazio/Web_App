# ========================================
# Setup & Design
# ========================================
# Modules
import base64
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
# Design 
# ~/.streamlit/config.toml
import streamlit as st
import streamlit_theme as stt
stt.set_theme({'primary': '#E694FF'})    
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
def get_ticker_data():
    tik = ticker_data(ticker_input)
    return tik

def py_data():
    ticker = get_ticker_data().df
    ticker = ticker.tz_localize('utc')
    return ticker
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
# Download Button
# ========================================
def download_link(object_to_download, download_filename, download_link_text):
    
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
# ========================================
# Launche App
# ========================================
# Create an instance of the app 

def main():

    st.title(":chart_with_upwards_trend: Mocca Application")

    st.sidebar.title("Model")
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
    
    fig = pf.tears.create_interesting_times_tear_sheet(df_pct, return_fig=True)
    
    st.write(fig)
    
    if st.button('Download Dataframe as CSV'):
        tmp_download_link = download_link(df, 'YOUR_DF.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()
    
