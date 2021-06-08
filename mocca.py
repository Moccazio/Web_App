# ========================================
# Setup & Design
# ========================================
import os
import numpy as np
from PIL import  Image
import streamlit as st
import streamlit as st
from streamlit.hashing import _CodeHasher
import time
import sqlite3
from sqlite3 import Error
import streamlit_theme as stt
stt.set_theme({'primary': '#E694FF'})    
# ========================================   
# Custom imports 
# ========================================   
from multipage import MultiPage
from pages import ticker, snp
# ========================================
# Main App
# ========================================

def main():
    st.set_page_config(layout="wide")
    app = MultiPage()

    display = Image.open('Logo.png')
    display = np.array(display)
    col1, col2 = st.beta_columns(2)
    col1.image(display, width = 100)
    col2.title("Mocca App")
    
    app.add_page("Company Ticker", ticker.app)
    #app.add_page("S&P 500 Ticker", snp.app)

    app.run()

if __name__ == '__main__':
    main()
    