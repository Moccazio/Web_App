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
                self.index = self.index.tz_localize('utc')
            else:
                self.df = self.df_ = self._ticker.history(start=start, end=end, auto_adjust=True)
                self.index = self.index.tz_localize('utc')
        except Exception as err:
            print(err)            
# ========================================                    
def read_nyse_ticker():
    nyse = pd.read_csv('ticker/nyse/nyse_ticker.csv', index_col = None)
    return nyse
# ========================================
# NYSE APP
# ========================================   
def app():
    st.title("New York Stock Exchange (NYSE)")  
    st.markdown("### select a ticker for analysis.") 
    nyse = read_nyse_ticker()  
    ticker = nyse['ticker'].sort_values().tolist()   
    nyse_ticker = st.selectbox('New York Stock Exchange',ticker) 
    nyse_data = pf_dat(nyse_ticker)
    data = nyse_data.df
    returns = data.Close.pct_change()
    st.table(pf.tears.create_interesting_times_tear_sheet(returns, return_fig=True))
