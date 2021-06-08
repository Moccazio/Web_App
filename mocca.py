# ========================================
# Setup & Design
# ========================================
import os
import time
import numpy as np
from PIL import  Image
import streamlit as st
import streamlit_theme as stt
stt.set_theme({'primary': '#E694FF'})    
# ========================================   
# Custom imports 
# ========================================   
#from multipage import MultiPage
# from pages import ticker
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
# ========================================
# Main App
# ========================================
def main():
    
    app = MultiPage()

    display = Image.open('Logo.png')
    display = np.array(display)
    col1, col2 = st.beta_columns(2)
    col1.image(display, width = 200)
    col2.subheader("Mocca Application")
    
    st.markdown("## Company Ticker")
    
    st.markdown('### search for a company ticker to start analysis.')
    
    ticker_input = st.text_input('Ticker')
    
    @st.cache()
    def load_data():
        stk = ticker_data(ticker_input)
        stk_history = stk.df.tz_localize('utc')
        stk_returns = stk_history.Close.pct_change().dropna()
        return  stk_history, stk_returns
    
    df_stk, df_pct = load_data()
    
    if ticker_input is not None:
        try: st.write(pf.timeseries.perf_stats(df_pct))
            
    
    if st.checkbox("Buy & Hold Return"):
        year = st.date_input("Buy-In Date (YYYY-MM-D)") 
        stock = ticker_data(ticker_input)
        stock_df = stock.df[year:]
        stock_df ['LogRets'] = np.log(stock_df['Close'] / stock_df['Close'].shift(1))
        stock_df['Buy&Hold_Log_Ret'] = stock_df['LogRets'].cumsum()
        stock_df['Buy&Hold_Return'] = np.exp(stock_df['Buy&Hold_Log_Ret'])
        font_1 = {'family' : 'Arial', 'size' : 12}
        fig2 = plt.figure()
        plt.style.use('dark_background')
        plt.title(ticker_input + ' Buy & Hold', fontdict = font_1)
        plt.plot(stock_df[['Buy&Hold_Return']])
        st.pyplot(fig2)
        st.dataframe(stock_df[['Buy&Hold_Rendite']])   
        
    if st.checkbox("Stress Event Analysis"):
        fig = pf.tears.create_interesting_times_tear_sheet(df_pct, return_fig=True)
        st.write(fig)

if __name__ == '__main__':
    main()    