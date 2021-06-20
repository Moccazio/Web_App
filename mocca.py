# ========================================
# Setup & Design
# ========================================
import os
import time
import numpy as np
from PIL import  Image
import streamlit as st 
from streamlit.hashing import _CodeHasher
import time
import boto3
import socket
import os
import getpass
import pandas as pd 
#
import sqlite3
from sqlite3 import Error
#
try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server
# ========================================   
# Custom imports 
# ========================================   
from multipage import MultiPage
from pages import app_dashboard, ticker
# ========================================
# Main App
# ========================================
st.set_page_config(layout="wide")
app = MultiPage()
display = Image.open('Logo.png')
display = np.array(display)
col1, col2 = st.beta_columns(2)
col1.image(display, width = 250)
col2.subheader("Mocca Web App")
#app.add_page("Dashboard", app_dashboard.app)
app.add_page("Ticker", ticker.app)

app.run()
    
    
