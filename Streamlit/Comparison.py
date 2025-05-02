import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def Comparison():
    st.title('Comparison')
    st.write('Comparison between red and white wine quality data')

    st.write('We are working with two different datasets that show data about white and red wine')
    st.write("First, we merge the two datasets together, which we can see here that it's now combined and have a type collumn, to see if it's red or white")

    # === Load Data ===
    data_folder = 'wine-data'
    dataset_white = os.path.join(data_folder, 'winequality-white.xlsx')
    dataset_red = os.path.join(data_folder, 'winequality-red.xlsx')

    try:
        # Load datasets and add type labels
        white_wine = pd.read_excel(dataset_white, skiprows=1)
        red_wine = pd.read_excel(dataset_red, skiprows=1)

        white_wine['type'] = 'white'
        red_wine['type'] = 'red'

        both_wine = pd.concat([white_wine, red_wine], ignore_index=True)
        both_wine['type_code'] = both_wine['type'].map({'red': 1, 'white': 2})
        numeric_cols = both_wine.select_dtypes(include='number').columns

        # === Preview Data ===
        preview_df = pd.concat([
            both_wine.head(4).assign(Row="Top"),
            both_wine.tail(4).assign(Row="Bottom")
        ]).drop(columns=["Row", "type_code"])

        st.subheader('Data Preview')
        st.dataframe(preview_df)

        # === Quality Summary ===
        st.subheader('Quality Summary by Wine Type')
        mean_quality = both_wine.groupby('type')['quality'].mean()
        max_quality = both_wine.groupby('type')['quality'].max()

        st.write("**Average Quality by Wine Type**")
        st.dataframe(mean_quality)

        st.write("**Highest Quality by Wine Type**")
        st.dataframe(max_quality)

        # === Correlation Analysis ===
        st.subheader('Correlation Insights')
        correlation_quality = both_wine[['alcohol', 'quality']].corr()
        correlation_density = both_wine[['alcohol', 'density']].corr()

        combined_corr = pd.concat(
            [correlation_quality, correlation_density],
            keys=["Alcohol vs Quality", "Alcohol vs Density"]
        )

        st.write("""
        - The **strongest correlation** in the dataset is between alcohol and quality.
        - The **weakest (most negative)** correlation is between alcohol and density, which indicates that alcohol content has little to no impact on wine density.
        """)
        st.dataframe(combined_corr)

        # === Scatterplot: Alcohol vs Quality ===
        st.subheader('Alcohol vs. Quality Scatterplot')
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x='alcohol', y='quality', hue='type', data=both_wine, alpha=0.5, ax=ax)
        ax.set_title('Alcohol vs. Quality')
        st.pyplot(fig)

        st.markdown("""
        - **Higher alcohol content** generally corresponds with **higher quality**. The scatterplot shows more clustering of data points in the upper-right portion of the graph.
        - As we move towards the right (higher alcohol content), the quality tends to increase, confirming a positive relationship between the two variables.
        - **In-depth observation**: For alcohol around 12–14%, many points correspond to quality scores of 7–8, while alcohol around 8–10% generally corresponds to quality scores of 3–5.
        """)

        # === Descriptive Statistics ===
        st.subheader("Descriptive Statistics")
        desc_stats = both_wine.describe().T.drop(columns='count')
        st.table(desc_stats)

        # === Q-Q Plots ===
        st.subheader("Q–Q Plots for Numerical Features")
        with st.expander("Show All Q–Q Plots"):
            for col in numeric_cols:
                fig = plt.figure()
                stats.probplot(both_wine[col].dropna(), dist="norm", plot=plt)
                plt.title(f"Q–Q Plot of {col}")
                st.pyplot(fig)

        st.markdown("""
        - **Normality assumption**: If the data points fall roughly on the 45° line, it indicates the data is normally distributed (e.g., pH, density, quality).
        - **Deviations from normality**: Volatile acidity, for example, shows an S-shaped curve, indicating a skewed distribution, which might require more analysis.
        - **What does this actually mean? Essentially if the data is normal it means there are no extreme outliers, no Michael Jordans (salary class example) in the data. So its assumed safe to work with. The opposite is true if the data isn't normal, and we may need a more in depth understanding of the data i.e. see who the Michael Jordan/Jordans in our data is/are.
        """)

        # === Boxplot: Quality by Type ===
        st.subheader("Wine Quality by Type")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.boxplot(x='type', y='quality', data=both_wine, ax=ax2)
        ax2.set_title('Wine Quality by Type')
        st.pyplot(fig2)

        st.write("""
        - **Wine Quality Distribution**: There is no **major difference** in quality between red and white wines. However, a more granular look reveals that **white wine** tends to have slightly higher quality scores on average.
        - This observation aligns with the **mean quality** values we calculated earlier.
        """)

        st.dataframe(mean_quality)

        # === Boxplot: Alcohol by Type ===
        st.subheader("Alcohol Content by Wine Type")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.boxplot(x='type', y='alcohol', data=both_wine, ax=ax3)
        ax3.set_title('Alcohol Content by Wine Type')
        st.pyplot(fig3)

        st.write("""
        - **Alcohol Content**: White wine typically has a **higher alcohol content** than red wine, which aligns with our earlier expectations.
        - This is confirmed by the mean alcohol values, which you can see below.
        """)

        mean_alcohol = both_wine.groupby('type')['alcohol'].mean()
        st.dataframe(mean_alcohol)

        return both_wine, numeric_cols

    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
