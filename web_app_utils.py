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
    start = '1900-01-01'
    end = dt.datetime.now()
    df = pdr.DataReader('^GSPC', 'yahoo', start, end)
    df.columns = ["S&P500"]
    return sp500

def get_vix_data():
    start = '1900-01-01'
    end = dt.datetime.now()
    df = pdr.DataReader('^VIX', 'yahoo', start, end)
    df.columns = ["VIX"]
    return vix

def get_dax_data():
    start = '1900-01-01'
    end = dt.datetime.now()
    df = pdr.DataReader('^GDAXI', 'yahoo', start, end)
    df.columns = ["DAX"]
    return dax

def get_mdax_data():
    start = '1900-01-01'
    end = dt.datetime.now()
    df = pdr.DataReader('^MDAXI', 'yahoo', start, end)
    df.columns = ["MDAX"]
    return mdax

def get_sdax_data():
    start = '1900-01-01'
    end = dt.datetime.now()
    df = pdr.DataReader('^SDAXI', 'yahoo', start, end)
    df.columns = ["SDAX"]
    return sdax

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
