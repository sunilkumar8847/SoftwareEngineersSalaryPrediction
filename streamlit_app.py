import streamlit as st
from predict_page import show_predic_page
from explore_page import show_explore_page

page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predic_page()
else:
    show_explore_page()
