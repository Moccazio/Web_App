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

# Custom imports 
from multipage import MultiPage
from pages import data, meta, redundant, data_visualize# import your pages here

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
def pyfolio_data(ticker):
    ticker = yf.Ticker(ticker)
    history = ticker.history('max')
    history.index = history.index.tz_localize('utc')
    return history
def get_vix_data():
    stk_price = ticker_data("^VIX").df
    df= stk_price.reset_index()
    df = df[["Date","Close"]]
    df = df.rename(columns = {"Date":"Datum","Close":"VIX"}) 
    df=df.set_index("Datum")
    return df
def get_dax_data():
    stk_price = Stock("^GDAXI").df
    df= stk_price.reset_index()
    df = df[["Date","Close"]]
    df = df.rename(columns = {"Date":"Datum","Close":"DAX"}) 
    df=df.set_index("Datum")
    return df
def get_mdax_data():
    stk_price = Stock("^MDAXI").df
    df= stk_price.reset_index()
    df = df[["Date","Close"]]
    df = df.rename(columns = {"Date":"Datum","Close":"MDAX"}) 
    df=df.set_index("Datum")
    return df
def get_sdax_data():
    stk_price = Stock("^SDAXI").df
    df= stk_price.reset_index()
    df = df[["Date","Close"]]
    df = df.rename(columns = {"Date":"Datum","Close":"SDAX"}) 
    df=df.set_index("Datum")
    return df
def read_dax_ticker():
    dax = pd.read_csv('index_stocks/DAX.csv', index_col='Index')
    return dax
def read_sp500_ticker():
    sp500 = pd.read_csv('index_stocks/GSPC.csv', index_col='Index')
    return sp500
def get_stock_data():
    stock = Stock(ticker_input)
    return stock
def dax_stock_data():
    stock = Stock(DAX_ticker)
    return stock
def snp_stock_data():
    stock = Stock(SNP_ticker)
    return stock
def get_quote_data():
    quote = pdr.get_quote_yahoo(ticker_input)
    return quote  
def get_option_data():
    options_df = Options_Chain(SNP_ticker)
    return options_df
# ========================================
# Prophet
# ========================================
def prophet_df(stk_price):
    df=stk_price.reset_index()
    df = df[["Date","Close"]] 
    df = df.rename(columns = {"Date":"ds","Close":"y"}) 
    return df
def predict_with_prophet():
    stk = get_stock_data()
    stk_df = stk.df["2010":]
    df = prophet_df(stk_df)
    return df
def predict_with_prophet_snp():
    stk = snp_stock_data()
    stk_df = stk.df["2010":]
    df = prophet_df(stk_df)
    return df
def predict_with_prophet_dax():
    stk = dax_stock_data()
    stk_df = stk.df["2010":]
    df = prophet_df(stk_df)
    return df
# ========================================
# Launche App
# ========================================
# Create an instance of the app 
app = MultiPage()
st.title(":chart_with_upwards_trend: Data Application")
# Add all your application here
app.add_page("Data", data_nyse.app)
#app.add_page("Change Metadata", meta.app)
#app.add_page("Data Analysis",data_visualize.app)
#app.add_page("Y-Parameter Optimization",redundant.app)
# The main app
app.run()