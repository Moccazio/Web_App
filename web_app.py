import pandas as pd
import  pandas_datareader as pdr
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
import datetime as dt
from datetime import datetime
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yahoo_fin.stock_info import get_quote_table
from deep_translator import GoogleTranslator
# ========================================  
# Company Data 
# ========================================    
def comma_format(number):
    if not pd.isna(number) and number != 0:
        return '{:,.0f}'.format(number)

def percentage_format(number):
    if not pd.isna(number) and number != 0:
        return '{:.1%}'.format(number)
    
class Company:
    def __init__(self, ticker):
        self.income_statement = si.get_income_statement(ticker)
        self.balance_sheet = si.get_balance_sheet(ticker)
        self.cash_flow_statement = si.get_cash_flow(ticker)
        self.inputs = self.get_inputs_df()

    def get_inputs_df(self):
        income_statement_list = ['totalRevenue', 'ebit', 
        'incomeBeforeTax', 'incomeTaxExpense'
        ]
        balance_sheet_list = ['totalCurrentAssets', 'cash',
        'totalCurrentLiabilities', 'shortLongTermDebt',
        'longTermDebt'
        ]
        balance_sheet_list_truncated = ['totalCurrentAssets', 'cash',
        'totalCurrentLiabilities', 'longTermDebt'
        ]
        balance_sheet_list_no_debt = ['totalCurrentAssets', 'cash',
        'totalCurrentLiabilities'
        ]

        cash_flow_statement_list = ['depreciation', 
        'capitalExpenditures'
        ]

        income_statement_df = self.income_statement.loc[income_statement_list]
        try:
            balance_sheet_df = self.balance_sheet.loc[balance_sheet_list]
        except KeyError:
            try:
                balance_sheet_df = self.balance_sheet.loc[balance_sheet_list_truncated]
            except KeyError:
                balance_sheet_df = self.balance_sheet.loc[balance_sheet_list_no_debt]
        cash_flow_statement_df = self.cash_flow_statement.loc[cash_flow_statement_list]

        df = income_statement_df.append(balance_sheet_df)
        df = df.append(cash_flow_statement_df)        

        columns_ts = df.columns
        columns_str = [str(i)[:10] for i in columns_ts]
        columns_dict = {}
        for i,f in zip(columns_ts, columns_str):
            columns_dict[i] = f
        df.rename(columns_dict, axis = 'columns', inplace = True)

        columns_str.reverse()
        df = df[columns_str]

        prior_revenue_list = [None]
        for i in range(len(df.loc['totalRevenue'])):
            if i != 0 and i != len(df.loc['totalRevenue']):
                prior_revenue_list.append(df.loc['totalRevenue'][i-1])

        df.loc['Einnahmen'] = prior_revenue_list
        df.loc['Einnahmenwachstum'] = (df.loc['totalRevenue'] - df.loc['Einnahmen']) / df.loc['Einnahmen']
        df.loc['EBIT-Marge'] = df.loc['ebit']/df.loc['totalRevenue'] 
        df.loc['Steueranteil'] = df.loc['incomeTaxExpense']/df.loc['incomeBeforeTax'] 
        df.loc['Nettogeldausgaben über Verkauf'] = (- df.loc['capitalExpenditures'] - df.loc['depreciation']) / df.loc['totalRevenue']
        try:
            df.loc['nwc'] = (df.loc['totalCurrentAssets'] - df.loc['cash']) - (df.loc['totalCurrentLiabilities'] - df.loc['shortLongTermDebt'])
        except KeyError:
            df.loc['nwc'] = (df.loc['totalCurrentAssets'] - df.loc['cash']) - (df.loc['totalCurrentLiabilities'])
        df.loc['NWC über Verkauf'] = df.loc['nwc']/df.loc['totalRevenue']
        try:
            df.loc['Nettoschulden'] = df.loc['shortLongTermDebt'] + df.loc['longTermDebt'] - df.loc['cash']
        except KeyError:
            try:
                df.loc['Nettoschulden'] = df.loc['longTermDebt'] - df.loc['cash']
            except KeyError:
                df.loc['Nettoschulden'] = - df.loc['cash']
        df = df[12:len(df)].drop('nwc')
        df['Historischerdurchschnitt'] = [df.iloc[i].mean() for i in range(len(df))]
        return df

    def get_free_cash_flow_forecast(self, parameter_list):
        df = pd.DataFrame(columns = [1, 2, 3, 4, 5])
        revenue_list = []
        for i in range(5):
            revenue_list.append(parameter_list[0] * (1 + parameter_list[1]) ** (i+1))
        df.loc['Revenues'] = revenue_list
        ebit_list = [i * parameter_list[2] for i in df.loc['Revenues']]
        df.loc['EBIT'] = ebit_list
        tax_list = [i * parameter_list[3] for i in df.loc['EBIT']]
        df.loc['Taxes'] = tax_list
        nopat_list = df.loc['EBIT'] - df.loc['Taxes']
        df.loc['NOPAT'] = nopat_list
        net_capex_list = [i * parameter_list[4] for i in df.loc['Revenues']]
        df.loc['Net capital expenditures'] = net_capex_list
        nwc_list = [i * parameter_list[5] for i in df.loc['Revenues']]
        df.loc['Changes in NWC'] = nwc_list
        free_cash_flow_list = df.loc['NOPAT'] - df.loc['Net capital expenditures'] - df.loc['Changes in NWC']
        df.loc['Free cash flow'] = free_cash_flow_list
        return df

    def discount_free_cash_flows(self, parameter_list, discount_rate, terminal_growth):
        free_cash_flow_df = self.get_free_cash_flow_forecast(parameter_list)
        df = free_cash_flow_df
        discount_factor_list = [(1 + discount_rate) ** i for i in free_cash_flow_df.columns]
        df.loc['Discount factor'] = discount_factor_list
        present_value_list = df.loc['Free cash flow'] / df.loc['Discount factor']
        df.loc['PV free cash flow'] = present_value_list
        df[0] = [0 for i in range(len(df))]
        df.loc['Sum PVs', 0] = df.loc['PV free cash flow', 1:5].sum()
        df.loc['Terminal value', 5] = df.loc['Free cash flow', 5] * (1 + terminal_growth) / (discount_rate - terminal_growth)
        df.loc['PV terminal value', 0] = df.loc['Terminal value', 5] / df.loc['Discount factor', 5]
        df.loc['Company value (enterprise value)', 0] = df.loc['Sum PVs', 0] + df.loc['PV terminal value', 0]
        df.loc['Net debt', 0] = parameter_list[-1]
        df.loc['Equity value', 0] = df.loc['Company value (enterprise value)', 0] - df.loc['Net debt', 0]
        equity_value = df.loc['Equity value', 0] 
        df = df.applymap(lambda x: comma_format(x))
        df = df.fillna('')
        column_name_list = range(6)
        df = df[column_name_list]
        return df, equity_value
# ========================================    
# Stock - Data
# ========================================    
class Stock:
    def __init__(self, ticker, start=None, end=None):

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
# Know your Options 
# ========================================    
def create_spread_percent(df):
    return (df.assign(spread_pct = lambda df: df.spread / df.ask))

def filter_by_moneyness(df, pct_cutoff=0.2):
    crit1 = (1-pct_cutoff)*df.strike < df.stockPrice
    crit2 = df.stockPrice < (1+pct_cutoff)*df.strike
    return (df.loc[crit1 & crit2].reset_index(drop=True))

def call_options(sym):
    stk = Options(sym)
    dates = stk.expiry_dates
    calls_merged = []
    for d in dates:
            calls_ = stk.get_call_data(expiry=d)
            calls_filtered = calls_[['Last', 'Bid', 'Ask', 'Chg', 'Vol', 'Open_Int', 'IV', 'Underlying_Price', 'Last_Trade_Date', 'Root']].reset_index()
            calls_filtered = calls_filtered [['Strike', 'Expiry', 'Type','Last', 'Bid', 'Ask', 'Chg', 'Vol', 'Open_Int', 'IV', 'Underlying_Price', 'Root']]
            calls_filtered.columns =  [['Strike', 'Expiry', 'Type','Last', 'Bid', 'Ask', 'Chg', 'Vol', 'Open_Int', 'IV', 'Underlying_Price', 'Symbol']]     
            calls_merged.append(calls_filtered)   
    calls_df = pd.concat(calls_merged, axis=0) 
    return calls_df

def put_options(sym):
    stk = Options(sym)
    dates = stk.expiry_dates
    puts_merged = []
    for d in dates:
            puts_ = stk.get_put_data(expiry=d)
            puts_filtered = puts_[['Last', 'Bid', 'Ask', 'Chg', 'Vol', 'Open_Int', 'IV', 'Underlying_Price', 'Last_Trade_Date', 'Root']].reset_index()
            puts_filtered = puts_filtered[['Strike', 'Expiry', 'Type','Last', 'Bid', 'Ask', 'Chg', 'Vol', 'Open_Int', 'IV', 'Underlying_Price', 'Root']]
            puts_filtered.columns =  [['Strike', 'Expiry', 'Type','Last', 'Bid', 'Ask', 'Chg', 'Vol', 'Open_Int', 'IV', 'Underlying_Price', 'Symbol']]    
            puts_merged.append(puts_filtered)
        
    puts_df = pd.concat(puts_merged, axis=0) 
    return puts_df

def options_chain(symbol):

    tk = yf.Ticker(symbol)
    
    info = get_quote_table(symbol)
    current_price = info["Quote Price"]

    # Expiration dates
    exps = tk.options

    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)

    # Bizarre error in yfinance that gives the wrong expiration date
    # Add 1 day to get the correct expiration date
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365
    
    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    options['spread'] = (options['ask'] - options['bid'])
    options = create_spread_percent(options)
    options['stockPrice'] = current_price
    options['intrinicValue'] = (options['strike'] - current_price)
    
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate'])
    
    options = filter_by_moneyness(options)
    options_filtered  = options [['stockPrice','contractSymbol', 'CALL', 'expirationDate', 'dte' ,'strike', 'lastPrice',  'bid', 'ask', 'mark', 'spread', 'spread_pct', 'volume', 'openInterest', 'impliedVolatility', 'inTheMoney', 'intrinicValue']]
    options_filtered.columns =  [[ 'Kurswert', 'OptionTicker', 'Call=1, Put=0', 'Einlösetermin', 'Ablaufdatum', 'Basispreis', 'Last', 'Bid', 'Ask', 'Mark', 'Spread', 'Spreadanteil', 'Volumen', 'Open Intrest', 'IV', 'im Geld', 'Substanzwert']]   
        

    return options_filtered 

def Options_Chain(ticker):
    opt_chain = options_chain(ticker)
    return opt_chain

def Option(ticker):
    _call = call_options(ticker)
    _put = put_options(ticker)
    _df =  pd.concat([_call, _put],  
                  keys = ['Call', 'Put'], ignore_index = True)
    #_df = pd.concat([_call, _put], axis=0)
    return _df

# ========================================     
# Economic Data Funktions
# ========================================   
def get_sp500_data():
    sp500 = Stock("^GSPC").df.Close
    return sp500

def get_dax_data():
    dax = Stock("^GDAXI").df.Close
    return dax

def get_wti_data():
    start = '1900-01-01'
    end = dt.datetime.now()
    df = pdr.DataReader('DCOILWTICO', 'fred', start, end)
    df.columns = ["WTI"]
    df = df.dropna()
    return df

def get_tres10_data():
    start = '1900-01-01'
    end = dt.datetime.now()
    df = pdr.DataReader('DGS10', 'fred', start, end)
    df.columns = ["Treasury Bill: 10-Jahre"]
    df = df.dropna()
    return df

def get_ggov_data():
    start = '1900-01-01'
    end = dt.datetime.now()
    df = pdr.DataReader('IRLTLT01DEM156N', 'fred', start, end)
    df.columns= ["Deutsche Staatsanleihen: 10-Jahre"]
    df = df.dropna()
    return df

def get_euinfl_data():
    start = '1900-01-01'
    end = dt.datetime.now()
    df = pdr.DataReader('PCPITOTLZGEMU', 'fred', start, end)
    df.columns= ["Inflationsrate in der Eurozone"]
    df = df.dropna()
    return df

def get_ginfl_data():
    start = '1900-01-01'
    end = dt.datetime.now()
    df = pdr.DataReader('FPCPITOTLZGDEU', 'fred', start, end)
    df.columns= ["Inflationsrate in Deutschland"]
    df = df.dropna()
    return df
                               
def get_unempger_data():
    start = '1900-01-01'
    end = dt.datetime.now()
    df = pdr.DataReader('LMUNRRTTDEM156S', 'fred', start, end)
    df.columns = ["Arbeitslosenquote in Deutschland"]
    df = df.dropna()
    return df
# ========================================
# Stock Data Funktions
# ========================================

def read_dax_ticker():
    dax = pd.read_csv('index_stocks/DAX.csv', index_col='Index')
    return dax

def read_sp500_ticker():
    sp500 = pd.read_csv('index_stocks/GSPC.csv', index_col='Index')
    return sp500

def read_internet_ticker():
    internet = pd.read_csv('koyfin_stocks/Internet_Stocks.csv')
    internet = internet[1:]
    return internet


def read_software_ticker():
    software = pd.read_csv('koyfin_stocks/SAAS_Stocks.csv')
    software = software[1:]
    return software

def get_stock_data():
    stock = Stock(ticker_input)
    return stock

def dax_stock_data():
    stock = Stock(DAX_ticker)
    return stock

def snp_stock_data():
    stock = Stock(SNP_ticker)
    return stock

def internet_stock_data():
    stock = Stock(internet_ticker)
    return stock

def software_stock_data():
    stock = Stock(software_ticker)
    return stock
    
def get_quote_data():
    quote = pdr.get_quote_yahoo(ticker_input)
    return quote 
  
def get_company_data():
    company = Company(ticker_input)
    return company

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

def predict_with_prophet_internet():
    stk = internet_stock_data()
    stk_df = stk.df["2010":]
    df = prophet_df(stk_df)
    return df

def predict_with_prophet_software():
    stk = software_stock_data()
    stk_df = stk.df["2010":]
    df = prophet_df(stk_df)
    return df

# ========================================
# Launche App
# ========================================
import streamlit as st
from streamlit.hashing import _CodeHasher

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server


def main():
    state = _get_state()
    pages = {
        "Dashboard": page_dashboard,
        "Einzelaktien": page_stocks
        "Indizes": page_settings,
        "Wirtschaft": page_eco
    }

    st.sidebar.title(":floppy_disk: Page states")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()
    
def page_dashboard(state):
    st.title(":chart_with_upwards_trend: Dashboard page")
    st.success("Zugang gewährt") 
    st.header('Die Aktien Gruppe')
    
    st.subheader('Aktienanalyse')
    st.markdown("Es muss ein Aktienticker eingegeben oder ausgewählt werden. Der Aktienticker ist der Kürzel mit dem die Aktie representativ gelistet ist, z.B. DPW.DE als Ticker für die Deutsche Post AG.")

    
def page_stocks(state):
    st.title(":chart_with_upwards_trend:  Einzelaktienanalyse")
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


        st.header('Handelsstrategien')    
        st.subheader('Aktienkurs')  
        st.subheader('Renditerechner')       
        
        if st.checkbox("Kaufen und Halten"):
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
        
        if st.checkbox("Prophet Kursprognose (Markov Chain Monte Carlo)"):
            df  = predict_with_prophet()
            fbp = Prophet(daily_seasonality = True)
            fbp.fit(df)
            fut = fbp.make_future_dataframe(periods=365) 
            forecast = fbp.predict(fut)
            fig2 = fbp.plot(forecast)
            st.pyplot(fig2)
            st.dataframe(forecast) 
            
def page_index(state):
    
    ticker_radio_1 = st.radio('Indizes', ('','S&P500', 'DAX'))       
    if ticker_radio_1 == 'S&P500':
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

        st.header('Handelsstrategien')    
    
        st.subheader('Renditerechner')       
        if st.checkbox("Kaufen und Halten"):
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
    
    
        st.subheader('Derivate')  
    
        if st.checkbox("Optionscheine"):
            options = get_option_data()
            st.dataframe(options)   
            
        
        
        st.subheader('Aktienkursprognose')  
    
        if st.checkbox("Markov Chain Monte Carlo"):
            df  = predict_with_prophet_snp()
            fbp = Prophet(daily_seasonality = True)
            fbp.fit(df)
            fut = fbp.make_future_dataframe(periods=365) 
            forecast = fbp.predict(fut)
            fig2 = fbp.plot(forecast)
            st.pyplot(fig2)
            st.dataframe(forecast)    
                
    
                
    if ticker_radio_1 == 'DAX':
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
        
        st.header('Handelsstrategien')     
        st.subheader('Renditerechner')       
    
        if st.checkbox("Kaufen und Halten"):
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
    
        
        
        
        
        
        st.subheader('Aktienkursprognose')  
    
        if st.checkbox("Markov Chain Monte Carlo"):
            df  = predict_with_prophet_dax()
            fbp = Prophet(daily_seasonality = True)
            fbp.fit(df)
            fut = fbp.make_future_dataframe(periods=365) 
            forecast = fbp.predict(fut)
            fig2 = fbp.plot(forecast)
            st.pyplot(fig2)
            st.dataframe(forecast)    
            
def page_eco(state):
    st.subheader('Wirtschaft')
    if st.checkbox("Wirschaftsindikatoren"):
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
    
        if (status == 'Arbeitslosenquote Deutschland'): 
            df=get_unempger_data()
            font_1 = {
             'family' : 'Arial',
                     'size' : 12
                }
            fig1 = plt.figure()
            plt.style.use('seaborn-whitegrid')
            plt.title('Arbeitslosenquote - Deutschland', fontdict = font_1)
            plt.plot(df)
            st.pyplot(fig1)
            st.dataframe(df)
    
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
    
        
        
        

    
    
    
    
    
