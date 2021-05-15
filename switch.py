import main
import noise
import dataexpo
import visual
import streamlit as st
import pandas as pd
st.set_page_config(page_title='The Machine Learning Algorithm Comparison App',
                   layout='wide')

PAGES = {
    "Noise detection": noise,
    "Algortihm selection": main,
    "Data Exploration": dataexpo,
    "Data visualization": visual
}
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input CSV file", type=["csv"])

value = st.selectbox("select option", options=list(PAGES.keys()))
page = PAGES[value]
if uploaded_file is not None:
    if value == "Algortihm selection":
        f=pd.read_csv("data/new.csv")
        if  f.empty:
            page.app1(uploaded_file)
        else:
            page.app1(f)
    elif value == "Data Exploration":
        f=pd.read_csv("data/new.csv")
        if f.empty:
            page.app3(uploaded_file)
        else:
            page.app3(f)
    elif value == "Data visualization":
        f=pd.read_csv("data/new.csv")
        if f.empty:
            page.app4(uploaded_file)
        else:
            page.app4(f)
    else:
        f = page.app2(uploaded_file)
        f.to_csv("data/new.csv",index=False)



else:
    st.warning("Upload your dataset")
