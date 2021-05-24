# ========================================
# Setup and Design
# ========================================
import streamlit as st 
import streamlit_theme as stt
stt.set_theme({'primary': '#F63366'})
import time
from web_app_utils import *
# ========================================
# Launche App
# ========================================

ticker_radio = st.sidebar.radio('Datenanalyse', ('Dashboard', 'Aktienticker', 'S&P500', 'DAX'))

if ticker_radio == 'Dashboard':
    st.title(":chart_with_upwards_trend: Dashboard")
    st.header('Die Aktiengruppe')
    st.markdown('...............................................................................................................................................................')
    st.markdown("“Heute kennt man von allem den Preis, von nichts den Wert.” (Oscar Wilde)")                
    st.markdown('...............................................................................................................................................................')
    st.markdown("S&P 500 Historische Wertentwicklung")        
    st.markdown('...............................................................................................................................................................')
    df=get_sp500_data()
    st.line_chart(df)
    st.markdown('...............................................................................................................................................................')
    st.header('Wirtschaft')
    status = st.radio("Wirschaftsindikatoren: ", ('','Crude Oil Prices: West Texas Intermediate (WTI)', 'US Treasury Anleihe: 10 Jahre',  'Deutsche Staatsanleihen: 10-Jahre', 'Inflationsrate Deutschland', 'Arbeitslosenquote Deutschland', 'Inflationsrate Eurozone')) 

    if (status == 'Crude Oil Prices: West Texas Intermediate (WTI)'): 
        df=get_wti_data()
        font_1 = {
            'family' : 'Arial',
              'size' : 12
                }
        fig1 = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.title('Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma', fontdict = font_1)
        plt.plot(df)
        st.pyplot(fig1)
        st.write(df)  
           
    if (status == 'US Treasury Anleihe: 10 Jahre'): 
        df=get_tres10_data()
        font_1 = {
         'family' : 'Arial',
         'size' : 12
          }
        fig1 = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.title('US Treasury Anleihe: 10 Jahre', fontdict = font_1)
        plt.plot(df)
        st.pyplot(fig1)
        st.write(df)  
    
    if (status == 'Deutsche Staatsanleihen: 10-Jahre'): 
        df=get_ggov_data()
        font_1 = {
        'family' : 'Arial',
             'size' : 12
              }
        fig1 = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.title('Deutsche Staatsanleihen: 10-Jahre', fontdict = font_1)
        plt.plot(df)
        st.pyplot(fig1)
        st.write(df)  
    
    if (status == 'Inflationsrate Deutschland'): 
        df=get_ginfl_data()
        font_1 = {
                     'family' : 'Arial',
                        'size' : 12
                 }
        fig1 = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.title('Inflationsrate - Deutschland', fontdict = font_1)
        plt.plot(df)
        st.pyplot(fig1)
        st.write(df)    
    
    if (status == 'Inflationsrate Eurozone'): 
        df=get_ginfl_data()
        font_1 = {
                    'family' : 'Arial',
                     'size' : 12
                 }
        fig1 = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.title('Inflationsrate Eurozone', fontdict = font_1)
        plt.plot(df)
        st.pyplot(fig1)
        st.dataframe(df)
        
if ticker_radio == 'Tickersuche':
    st.subheader('Aktienanalyse')
    st.markdown("Es muss ein Aktienticker eingegeben oder ausgewählt werden. Der Aktienticker ist der Kürzel mit dem die Aktie representativ gelistet ist, z.B. DPW.DE als Ticker für die Deutsche Post AG.")
    ticker_input = st.text_input('Ticker')
    status_radio = st.radio('Suche anklicken um zu starten.', ('Eingabe', 'Suche'))          
    
    
    if status_radio == 'Suche':
        
        stock_i = yf.Ticker(ticker_input)
        info = stock_i.info 
        to_translate_1 = info['sector']
        to_translate_2 = info['industry']
        translated_1 = GoogleTranslator(source='auto', target='de').translate(to_translate_1)
        translated_2 = GoogleTranslator(source='auto', target='de').translate(to_translate_2)
        st.subheader(info['longName'])
        st.markdown('** Sektor **: ' + translated_1)
        st.markdown('** Industrie **: ' + translated_2)
        
        st.header('Datenanalyse')
        
        if st.checkbox("Finanzkennzahlen"):
            company = get_company_data()
            st.dataframe(company.inputs)  
        stock = get_stock_data()    
        close = stock.df.Close
        if st.checkbox("Graphischer Kursverlauf"):
            font_1 = {
                    'family' : 'Arial',
                         'size' : 12
                    }
            fig1 = plt.figure()
            plt.style.use('seaborn-whitegrid')
            plt.title(ticker_input + ' Kursverlauf', fontdict = font_1)
            plt.plot(close)
            st.pyplot(fig1)      
        
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
            plt.style.use('seaborn-whitegrid')
            plt.title(ticker_input + ' Kaufen und Halten', fontdict = font_1)
            plt.plot(stock_df[['Buy&Hold_Rendite']])
            st.pyplot(fig2)
            st.dataframe(stock_df[['Buy&Hold_Rendite']])
                  
        st.subheader('Aktienkursprognose') 
        
        if st.checkbox("Aktienkursprognose (Markov Chain Monte Carlo)"):
            df  = predict_with_prophet()
            fbp = Prophet(daily_seasonality = True)
            fbp.fit(df)
            fut = fbp.make_future_dataframe(periods=365) 
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
        plt.title(SNP_ticker + ' Kaufen und Halten', fontdict = font_1)
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
        

