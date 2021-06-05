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
def read_nyse_ticker():
    nyse = pd.read_csv('ticker/nyse/nyse_ticker.csv', index_col = None)
    return nyse
# ========================================   
def app():
    st.title("New York Stock Exchange (NYSE)")  
    st.markdown("### select a ticker for analysis.") 
    nyse = read_nyse_ticker()  
    ticker = nyse['ticker'].sort_values().tolist()   
    nyse_ticker = st.selectbox('New York Stock Exchange',ticker) 
    nyse_data = ticker_data(nyse_ticker)
    data = nyse_data.df
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
    columns = []
    columns = utils.genMetaData(data)
    columns_df = pd.DataFrame(columns, columns = ['column_name', 'type'])
    st.markdown("**Column Name**-**Type**")
    for i in range(columns_df.shape[0]):
        st.write(f"{i+1}. **{columns_df.iloc[i]['column_name']}** - {columns_df.iloc[i]['type']}")
    st.markdown("""The above are the automated column types detected by the application in the data.\
    In case you wish to change the column types, head over to the **Column Change** section. """)  