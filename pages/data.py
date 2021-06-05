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
@st.cache
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
@st.cache                     
def read_nyse_ticker():
    sp500 = pd.read_csv('ticker/nyse/nyse_ticer.csv').ticker
    return sp500
# ========================================   
def nyse_stock_data():
    nyse_data = ticker_data(nyse_ticker).df
    return nyse_data
# ========================================   
def get_ticker_data():
    ticker_data = ticker_data(ticker_input).df
    return ticker_data
# ========================================   
@st.cache
def app():
    st.title("Stock Ticker")
    ticker_radio = st.radio('Ticker', ('', 'NYSE', 'Ticker'))   
    st.markdown("### select a ticker for analysis.") 
    st.write("\n")   
    
    if ticker_radio == 'NYSE':
        nyse = read_nyse_ticker()  
        ticker_nyse= nyse['ticker'].sort_values().tolist()   
        nyse_ticker = st.selectbox('New York Stock Exchange',ticker_nyse) 
        global data
        if nyse_ticker is not None:
            data = nyse_stock_data()
    if ticker_radio == 'Ticker':
        ticker_input = st.text_input('Ticker')
        global data
        try:
            data = get_ticker_data()
            
    st.dataframe(data)     
    utils.getProfile(data)
    st.markdown("<a href='output.html' download target='_blank' > Download profiling report </a>",unsafe_allow_html=True)
    data.to_csv('data/main_data.csv', index=False)
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
    columns = []
    columns = utils.genMetaData(data) 
    columns_df = pd.DataFrame(columns, columns = ['column_name', 'type'])
    columns_df.to_csv('data/metadata/column_type_desc.csv', index = False)
    st.markdown("**Column Name**-**Type**")
    for i in range(columns_df.shape[0]):
        st.write(f"{i+1}. **{columns_df.iloc[i]['column_name']}** - {columns_df.iloc[i]['type']}")
    st.markdown("""The above are the automated column types detected by the application in the data.\
    In case you wish to change the column types, head over to the **Column Change** section. """)    