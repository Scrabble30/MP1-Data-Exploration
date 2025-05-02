import os
import pandas as pd
import streamlit as st

def WhiteWine():
    st.title('White Wine')
    st.write('tadadadada')

    data_folder = 'wine-data'
    dataset_path = os.path.join(data_folder,'winequality-white.xlsx')

    try:
        df = pd.read_excel(dataset_path, skiprows=1)
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return
    
    with st.expander("White Wine data"):
        st.dataframe(df.head())

