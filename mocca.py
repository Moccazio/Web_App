# ========================================
# Setup & Design
# ========================================
import os
import time
import numpy as np
from PIL import  Image
import streamlit as st 
# ========================================   
# Custom imports 
# ========================================   
from multipage import MultiPage
from pages import ticker, nasdaq
# ========================================
# Main App
# ========================================
def main():
    st.set_page_config(layout="wide")
    app = MultiPage()
    display = Image.open('Logo.png')
    display = np.array(display)
    col1, col2 = st.beta_columns(2)
    col1.image(display, width = 200)
    col2.subheader("Mocca Web App")
    
    app.add_page("Ticker", ticker.app)
    app.add_page("NASDAQ", nasdaq.app)

    app.run()

if __name__ == '__main__':
    main()    