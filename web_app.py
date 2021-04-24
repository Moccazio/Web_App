import streamlit as st
from multiapp import MultiApp
from apps import home, stock # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Einzelaktien", stock.app)

# The main app
app.run()