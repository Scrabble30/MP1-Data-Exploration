import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def RedWine():
    st.title('Red Wine Analysis')
    st.write('In-depth exploration of red wine quality characteristics')

    data_folder = 'wine-data'
    dataset_red = os.path.join(data_folder, 'winequality-red.xlsx')

    try:
        red_wine = pd.read_excel(dataset_red, skiprows=1)
        red_wine['type'] = 'red'
        numeric_cols = red_wine.select_dtypes(include='number').columns

        st.subheader('Data Preview')
        st.dataframe(red_wine.head(10))

        # Quality Summary
        st.subheader('Quality Statistics')
        mean_quality = red_wine['quality'].mean()
        max_quality = red_wine['quality'].max()
        st.write(f"**Average Quality**: {mean_quality:.2f}")
        st.write(f"**Max Quality**: {max_quality}")

        # Correlation Matrix
        st.subheader('Correlation with Quality')
        corr_matrix = red_wine[numeric_cols].corr()
        corr_quality = corr_matrix['quality'].sort_values(ascending=False)
        st.dataframe(corr_quality)

        # Heatmap of Correlation Matrix 
        st.subheader("Heatmap of Correlation Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax_corr)
        ax_corr.set_title("Red Wine Feature Correlation Heatmap")
        st.pyplot(fig_corr)

        st.write("As seen in both white wine dataset and in both white and red merged we see that" \
        "alcohol has the strongest positive correlation with quality, " \
        "while volatile acidity and density have negative correlations. A note of value here is that the Pearson Coefficient" \
        "is a bit higher for redwine (0.48) than white (0.44) implying that alcohol usually affects redwine more positively" \
        "than white wine.")

        # Scatterplot: Alcohol vs Quality
        st.subheader('Alcohol vs Quality')
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x='alcohol', y='quality', data=red_wine, alpha=0.6, ax=ax1)
        ax1.set_title('Alcohol vs Quality (Red Wine)')
        st.pyplot(fig1)

        # Boxplot: Quality
        st.subheader('Quality Distribution')
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='quality', data=red_wine, ax=ax2)
        ax2.set_title('Red Wine Quality Distribution')
        st.pyplot(fig2)

        # Q–Q Plots
        st.subheader("Q–Q Plots for Red Wine Features")
        with st.expander("Show Q–Q Plots"):
            for col in numeric_cols:
                fig = plt.figure()
                stats.probplot(red_wine[col].dropna(), dist="norm", plot=plt)
                plt.title(f"Q–Q Plot of {col}")
                st.pyplot(fig)

        # Descriptive Statistics
        st.subheader("Descriptive Statistics")
        st.table(red_wine.describe().T.drop(columns='count'))

    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
