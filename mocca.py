# ========================================
# Setup & Design
# ========================================
import os
import numpy as np
from PIL import  Image
import streamlit as st
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
app = MultiPage()

display = Image.open('Logo.png')
display = np.array(display)
col1, col2 = st.beta_columns(2)
col1.image(display, width = 400)
col2.title("Mocca Application")

app.add_page("Ticker Search", ticker.app)
app.add_page("S&P 500 Ticker", snp.app)

app.run()