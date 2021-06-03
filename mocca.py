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
def get_sp500_data():
    ticker = yf.Ticker('^GSPC')
    history = ticker.history('max')
    history.index = history.index.tz_localize('utc')
    return history

def get_vix_data():
    stk_price = Stock("^VIX").df
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
ticker_radio = st.sidebar.radio('Seite', ('Dashboard', 'Aktienanalyse'))

if ticker_radio == 'Dashboard':
    st.title(":chart_with_upwards_trend:")
    st.markdown('...............................................................................................................................................................')
    st.markdown("“")                
    st.markdown('...............................................................................................................................................................')
    st.subheader("S&P 500 Index")        
    df_1 = get_sp500_data()
    returns = df_1.Close.pct_change()
    times_tear_shit = pf.plotting.create_interesting_times_tear_sheet(returns)
    st.write(times_tear_shit)
        
        
    st.subheader("CBOE Volatility Index Historische Wertentwicklung")
    st.markdown('Der Volatility Index (VIX) ist eine Zahl, die von den Preisen der Optionsprämie im S&P 500-Index abgeleitet ist. Liegt der VIX unter 20, wird für den Markt ein eher  gesundes und risikoarmes Umfeld prognostiziert.\
    Wenn der VIX jedoch zu tief fällt, ist dies Ausdruck von stark optimistisch gestimmten Investoren. Wenn der VIX auf über 20 steigt, dann beginnt die Angst in den Markt einzutreten und es wird ein höheres Risikoumfeld\
    prognostiziert.')
    df_2 = get_vix_data()
    st.line_chart(df_2)
    st.table(df_2.tail())
    
    
    
    st.markdown('...............................................................................................................................................................')
    st.subheader("DAX Historische Wertentwicklung")
    df_3 = get_dax_data()
    st.area_chart(df_3)
    st.subheader("MDAX Historische Wertentwicklung")
    df_4 = get_mdax_data()
    st.area_chart(df_4)
    st.subheader("SDAX Historische Wertentwicklung")
    df_5 = get_sdax_data()
    st.area_chart(df_5)
    st.markdown('...............................................................................................................................................................')
        
if ticker_radio == 'Aktienanalyse':
    st.subheader('Aktienanalyse')
    st.markdown("Es muss ein Aktienticker eingegeben oder ausgewählt werden. Der Aktienticker ist der Kürzel mit dem die Aktie representativ gelistet ist, z.B. DPW.DE als Ticker für die Deutsche Post AG.")
    ticker_input = st.text_input('Ticker')
    st.subheader('Chart')
    stock = get_stock_data().df    
    chart_data_stk = pd.DataFrame({ticker_input: stock.Close})
    st.area_chart(chart_data_stk)
        
    if st.checkbox("Renditerechner"):
        year = st.date_input("Datum an den die Aktie gekauft wurde (YYYY-MM-D)") 
        stock = get_stock_data()
        stock_df = stock.df[year:]
        stock_df ['LogRets'] = np.log(stock_df['Close'] / stock_df['Close'].shift(1))
        stock_df['Buy&Hold_Log_Ret'] = stock_df['LogRets'].cumsum()
        stock_df['Buy&Hold_Rendite'] = np.exp(stock_df['Buy&Hold_Log_Ret'])
        font_1 = {
                'family' : 'Arial',
                 'size' : 12
                        }
        fig2 = plt.figure()
        plt.style.use('dark_background')
        plt.title(ticker_input + ' Buy & Hold Rendite', fontdict = font_1)
        plt.plot(stock_df[['Buy&Hold_Rendite']])
        st.write(fig2)
        st.dataframe(stock_df[['Buy&Hold_Rendite']])
                  
    st.subheader('Aktienkursprognose') 
    if st.checkbox("Aktienkursprognose (Markov Chain Monte Carlo)"):
        df  = predict_with_prophet()
        fbp = Prophet(daily_seasonality = True)
        fbp.fit(df)
        fut = fbp.make_future_dataframe(periods=252) 
        forecast = fbp.predict(fut)
        fig2 = fbp.plot(forecast)
        st.pyplot(fig2)
        st.dataframe(forecast)   
            

if ticker_radio == 'S&P500':
    snp500 = read_sp500_ticker()  
    ticker_snp = snp500['ticker'].sort_values().tolist()      
    SNP_ticker = st.selectbox(
    'S&P 500 Ticker auswählen',
      ticker_snp)  
    st.subheader("Ticker Info")
    stock_i = yf.Ticker(SNP_ticker)
    info = stock_i.info 
    to_translate_1 = info['sector']
    to_translate_2 = info['industry']
    translated_1 = GoogleTranslator(source='auto', target='de').translate(to_translate_1)
    translated_2 = GoogleTranslator(source='auto', target='de').translate(to_translate_2)
    st.subheader(info['longName'])
    st.markdown('** Sektor **: ' + translated_1)
    st.markdown('** Industrie **: ' + translated_2)
    st.header('Datenanalyse')
    
        
    stock = snp_stock_data()    
    close = stock.df.Close
    if st.checkbox("Graphischer Kursverlauf"):
        font_1 = {
                    'family' : 'Arial',
                         'size' : 12
                    }
        fig1 = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.title(SNP_ticker + ' Kursverlauf', fontdict = font_1)
        plt.plot(close)
        st.pyplot(fig1)

    if st.checkbox("Optionscheine"):
        options = get_option_data()
        st.dataframe(options)
        
    if st.checkbox("Renditerechner"):
        year = st.date_input("Datum an den die Aktie gekauft wurde (YYYY-MM-D)") 
        stock = snp_stock_data()
        stock_df = stock.df[year:]
        stock_df ['LogRets'] = np.log(stock_df['Close'] / stock_df['Close'].shift(1))
        stock_df['Buy&Hold_Log_Ret'] = stock_df['LogRets'].cumsum()
        stock_df['Buy&Hold_Rendite'] = np.exp(stock_df['Buy&Hold_Log_Ret'])
        font_1 = {
                'family' : 'Arial',
                 'size' : 12
                        }
        fig2 = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.title(SNP_ticker + ' Buy & Hold', fontdict = font_1)
        plt.plot(stock_df[['Buy&Hold_Rendite']])
        st.pyplot(fig2)
        st.dataframe(stock_df[['Buy&Hold_Rendite']])    
        
    if st.checkbox("Aktienkursprognose (Markov Chain Monte Carlo)"):
        df  = predict_with_prophet_snp()
        fbp = Prophet(daily_seasonality = True)
        fbp.fit(df)
        fut = fbp.make_future_dataframe(periods=365) 
        forecast = fbp.predict(fut)
        fig2 = fbp.plot(forecast)
        st.pyplot(fig2)
        st.dataframe(forecast)        
                
if ticker_radio == 'DAX':
    dax_ticker = read_dax_ticker()  
    ticker_dax = dax_ticker['ticker'].sort_values().tolist()  
    DAX_ticker = st.selectbox(
    'DAX Aktien',
      dax_ticker)  
    st.subheader("Ticker Info")
    stock_i = yf.Ticker(DAX_ticker)
    info = stock_i.info 
    to_translate_1 = info['sector']
    to_translate_2 = info['industry']
    translated_1 = GoogleTranslator(source='auto', target='de').translate(to_translate_1)
    translated_2 = GoogleTranslator(source='auto', target='de').translate(to_translate_2)
    st.subheader(info['longName'])
    st.markdown('** Sektor **: ' + translated_1)
    st.markdown('** Industrie **: ' + translated_2)
    st.header('Datenanalyse')
    
    stock = dax_stock_data()   
    close = stock.df.Close
    if st.checkbox("Graphischer Kursverlauf"):
        font_1 = {
                    'family' : 'Arial',
                         'size' : 12
                    }
        fig1 = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.title(DAX_ticker + ' Kursverlauf', fontdict = font_1)
        plt.plot(close)
        st.pyplot(fig1)
    
    if st.checkbox("Renditerechner"):
        year = st.date_input("Datum an den die Aktie gekauft wurde (YYYY-MM-D)") 
        stock = dax_stock_data()
        stock_df = stock.df[year:]
        stock_df ['LogRets'] = np.log(stock_df['Close'] / stock_df['Close'].shift(1))
        stock_df['Buy&Hold_Log_Ret'] = stock_df['LogRets'].cumsum()
        stock_df['Buy&Hold_Rendite'] = np.exp(stock_df['Buy&Hold_Log_Ret'])
        font_1 = {
                'family' : 'Arial',
                 'size' : 12
                        }
        fig2 = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.title(DAX_ticker + ' Kaufen und Halten', fontdict = font_1)
        plt.plot(stock_df[['Buy&Hold_Rendite']])
        st.pyplot(fig2)
        st.dataframe(stock_df[['Buy&Hold_Rendite']])    
    
    if st.checkbox("Aktienkursprognose (Markov Chain Monte Carlo)"):
        df  = predict_with_prophet_dax()
        fbp = Prophet(daily_seasonality = True)
        fbp.fit(df)
        fut = fbp.make_future_dataframe(periods=365) 
        forecast = fbp.predict(fut)
        fig2 = fbp.plot(forecast)
        st.pyplot(fig2)
        st.dataframe(forecast)     
            

st.markdown('...............................................................................................................................................................')
st.markdown("*Für die aufgeführten Inhalte kann keine Gewährleistung für die Vollständigkeit, Richtigkeit und Genauigkeit übernommen werden.*")
st.markdown('..............................................................................................................................................................')        
#st.sidebar.markdown('<a href="mailto:">Contact me !</a>', unsafe_allow_html=True)    
        

