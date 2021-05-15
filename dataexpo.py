import pandas as pd
from sklearn.impute import SimpleImputer
import streamlit as st
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import random
import main
import numpy as np
from sklearn.preprocessing import LabelEncoder
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

import warnings


def app3(uploaded_file):
    warnings.filterwarnings("ignore")
    df = pd.read_csv(uploaded_file)
    pr = ProfileReport(df, explorative=True)
    st.header('**Pandas Profiling Report(Exploratory data Analysis)**')
    st_profile_report(pr)
def app3(df):
    warnings.filterwarnings("ignore")
    pr = ProfileReport(df, explorative=True)
    st.header('**Pandas Profiling Report(Exploratory data Analysis)**')
    st_profile_report(pr)
