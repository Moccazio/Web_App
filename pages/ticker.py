import collections
from numpy.core.defchararray import lower
import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
import streamlit.components.v1 as components 
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
# St APP
# ========================================   
def app():
    st.title("Ticker Data Analysis")  
    st.markdown("### enter a ticker to start analysis.") 
    ticker_input = st.text_input('Ticker')
    temp_ticker = yf.Ticker(ticker_input)
    history = temp_ticker.history('max')
    history.index = history.index.tz_localize('utc')
    returns = history.Close.pct_change().dropna()
    st.write(pf.plotting.plot_annual_returns(returns))
    st.write(pf.plotting.plot_monthly_returns_heatmap(returns))
    st.write(pf.plotting.plot_rolling_sharpe(returns))
    st.write(pf.plotting.plot_drawdown_underwater(returns))
    st.write(pf.tears.create_interesting_times_tear_sheet(returns, return_fig=True))
