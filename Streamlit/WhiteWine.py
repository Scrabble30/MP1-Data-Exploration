import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def WhiteWine():
    st.title('White Wine Dataset Exploration')

    data_folder = 'wine-data'
    dataset_path = os.path.join(data_folder, 'winequality-white.xlsx')

    try:
        df = pd.read_excel(dataset_path, skiprows=1)
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return

    with st.expander("Raw Data"):
        st.dataframe(df.head())

    with st.expander("Missing Values"):
        st.write(df.isna().sum())

    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    st.subheader("Quality Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='quality', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Distributions")
    numeric_cols = df.select_dtypes(include='number').columns
    selected_feature = st.selectbox("Select a feature for histogram", numeric_cols)
    fig2, ax2 = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Boxplot by Quality")
    selected_feature_box = st.selectbox("Select a feature for boxplot", numeric_cols, key="boxplot")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='quality', y=selected_feature_box, data=df, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)
