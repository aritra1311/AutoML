import pandas as pd
from sklearn.impute import SimpleImputer
import streamlit as st
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import random
import main
import noise
import dataexpo
import visual
import numpy as np
from sklearn.preprocessing import LabelEncoder
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

#data = pd.read_csv("data.csv").iloc[:, :14]
# print(data.describe())

# detect column with noises



def app2(uploaded_file):

    st.header("Missing value and Noise detection")

    # missing value

    def noise(data):
        l = len(data.columns)
        col_list = list(data.columns)
        noise = data.isnull().sum()
        noisy_col = []
        for i in range(l):
            if(noise[i] != 0):
                noisy_col.append(col_list[i])
        return noisy_col

    # correct those column

    def correct(data, noisy_col):
        imputer = SimpleImputer(strategy="median")

        for i in range(len(noisy_col)):
            imputer.fit(data[[noisy_col[i]]])
            data[[noisy_col[i]]] = imputer.transform(data[[noisy_col[i]]])
        return data

    # uneccesary data columns

    def unecessary_col(data, label):
        l = len(data.columns)
        col_list = list(data.columns)
        index = l-1
        for i in range(l):
            if label == col_list[i]:
                index = i
        corr_value = []
        unused_col = []
        for i in range(l):
            cor_val = data[col_list[index]].corr(data[col_list[i]])
            corr_value.append(cor_val)
        for i in range(l):
            if(corr_value[i] < 0.05 and corr_value[i] > -0.05):
                unused_col.append(col_list[i])
        if(len(unused_col) == 0):
            return "No unnecessary column", data
        return unused_col, data

    # download the correct dataset

    def filedownload(df, filename):
        csv = df.to_csv(index=None)
        # strings <-> bytes conversions
        filename = filename+".csv"
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
        return href

    # column data noise reduction

    def noise_reduction(new_data, user_input):
        Q1 = new_data[user_input].quantile(0.25)
        Q3 = new_data[user_input].quantile(0.75)

        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5*IQR
        upper_limit = Q3 + 1.5*IQR
        mid = (lower_limit + upper_limit)/2
        for i in range(len(new_data)):
            curr_value = new_data.iloc[i][user_input]

            if(curr_value < lower_limit or curr_value > upper_limit):
                new_data.at[i, user_input] = random.randint(
                    int(lower_limit), int(upper_limit))
        return new_data

    # noise present or not

    def noise_check(new_data, user_input):
        Q1 = new_data[user_input].quantile(0.25)
        Q3 = new_data[user_input].quantile(0.75)

        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5*IQR
        upper_limit = Q3 + 1.5*IQR
        check = False
        stg = "Noise not present"
        #mid = (lower_limit + upper_limit)/2
        for i in range(len(new_data)):
            curr_value = new_data.iloc[i][user_input]

            if(curr_value < lower_limit or curr_value > upper_limit):
                check = True
            if(check):
                stg = "Noise present"
                break

        return stg

    def drop_col(data, string):
        for i in range(len(string)):
            data = data.drop([string[i]], axis=1)
        return data

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df)
        df = df.loc[:, ~df.columns.str.match("Unnamed")]
        for label, content in df.items():
            if pd.api.types.is_string_dtype(content):
                df[label] = content.astype("category").cat.codes + 1
        st.markdown("**New prepared dataset**")
        st.write(df)
        noise_col = noise(df)
        if(len(noise_col) == 0):
            st.success("No missing value present")
        else:
            st.warning("These column contain missing values :\n")
            for i in range(len(noise_col)):
                st.error(noise_col[i])
        col_list = list(df.columns)
        new_data = correct(df, noise_col)
        st.markdown(filedownload(new_data, "Corrected_dataset"),
                    unsafe_allow_html=True)

        # missing values case ends

        label = st.selectbox("Enter Your label name", col_list)
        if label != "":

            string, new_data = unecessary_col(df, label)
            if(string == "No unnecessary column"):
                st.success("No unnecessary column")
            else:
                st.warning("Unecessary columns are : ")
                for i in range(len(string)):
                    st.error(string[i])

            # Drop unnecessary columns
            if string != "No unnecessary column":
                drop = st.checkbox("Drop unncessary column")
                if drop:
                    new_data = drop_col(new_data, string)
                    st.write(new_data)
                    st.markdown(filedownload(new_data, "New_corrected_dataset"),
                                unsafe_allow_html=True)
            agree = st.checkbox("Detect column noise")
            col_list = list(df.columns)
            if agree:
                a = 0
                st.markdown("Dataset anamolies")

                user_input = st.selectbox("Select the column", col_list)
                if user_input != "":
                    if user_input not in col_list:
                        st.write("Enter valid column")
                    else:
                        # user_input = int(user_input)
                        plt.figure(figsize=(9, 3))
                        sns.set_theme(style="whitegrid")
                        ax = sns.boxplot(x=new_data[user_input])
                        # ax.set(ylim=(0, 1))
                        # plt.xticks(rotation=90)
                        st.pyplot(plt)
                        stg = noise_check(new_data, user_input)
                        if(stg == "Noise not present"):
                            st.success(stg)
                        else:
                            st.error(stg)
                        a = 1
                if a == 1:
                    # noise remove part
                    if st.button("Reduction of noise in that column"):
                        noise_free_data = noise_reduction(new_data, user_input)
                        st.write(new_data)
                        st.markdown(filedownload(
                            noise_free_data, "Noise_reduced_dataset"), unsafe_allow_html=True)
                    if st.button("Reduce noise from all the columns"):
                        for col in col_list:
                            noise_free_data = noise_reduction(
                                new_data, user_input)
                            new_data = noise_free_data
                        st.write(new_data)
                        st.markdown(filedownload(
                            noise_free_data, "Noise_reduced_dataset"), unsafe_allow_html=True)
        st.markdown("Custom Removing Columns")
        col_list = list(new_data.columns)
        l=len(col_list)
        i=0
        c = st.slider('Select How many times you want to drop columns', 0, l, 1)
        while(c>0):
            c1=list(new_data.columns)
            l1 = st.selectbox("Select the column to be dropped", col_list, key=c)
            c=c-1
            if(l1 not in c1):
                st.write("Column already dropped")
            else:
                new_data=new_data.drop(l1, axis=1)
                st.write(new_data)
                st.markdown(filedownload(new_data, "New_corrected_dataset"),
                  unsafe_allow_html=True)
    return new_data
        #st.header("Please Upload the new dataset again generated for next stages")



# noisy_col = noise(data)

# noise_free_data = correct(data, noisy_col)
# print(unecessary_col(noise_free_data))
