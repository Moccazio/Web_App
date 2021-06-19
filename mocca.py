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
from pages import ticker, snp, app_dashboard
# ========================================
# Main App
# ========================================
DATABASE_FILE_LOCATION = os.getcwd()+"\pythonsqlite.db" 

TABLE_DIC = {'stocks':'stocks','stock_trans':'stock_transaction'} 



def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)

        if (conn.execute("SELECT name FROM sqlite_master").fetchall() not in TABLE_DIC.values()):
            print("Creating "+ str(TABLE_DIC.values()) +" Table")

        conn.execute("""CREATE TABLE IF NOT EXISTS """ + TABLE_DIC['stock_trans'] + """ (    
                    Stock                       TEXT     PRIMARY KEY     NOT NULL,
                    Bought_Price                REAL                     NOT NULL,
                    Currency                    TEXT                     NOT NULL,
                    Fees                        REAL                     NOT NULL,
                    Quantity                    REAL                     NOT NULL);""")
    
        conn.execute("""CREATE TABLE IF NOT EXISTS """ + TABLE_DIC['stocks'] + """ (
                    Stock                       TEXT     PRIMARY KEY     NOT NULL,
                    Bought_Price_Avg            REAL                     NOT NULL,
                    Currency                    TEXT                     NOT NULL,
                    Fees                        REAL                     NOT NULL,
                    Quantity                    REAL                     NOT NULL);""")

        print("Successfully created "+ str(TABLE_DIC.values()) +" Table")

    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state    

def main():
    
    st.set_page_config(layout="wide")
    
    app = MultiPage()

    display = Image.open('Logo.png')
    display = np.array(display)
    col1, col2 = st.beta_columns(2)
    col1.image(display, width = 300)
    col2.subheader("Mocca Application")
    
    app.add_page("Company Ticker", ticker.app)
    app.add_page("Dashboard", app_dashboard.app)

    #Removing and add pages 

    create_connection(DATABASE_FILE_LOCATION) 

    st.sidebar.title(":floppy_disk:")

    app.run()
    
    
    

if __name__ == '__main__':
    main()    