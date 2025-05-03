import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
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

        # === Correlation Matrix Analysis for Wine Dataset ===

        # Compute the correlation matrix, excluding 'type_code'
        corr_matrix = both_wine.drop(columns='type_code').corr(numeric_only=True)

        st.subheader("Correlation Matrix (Numeric Features)")
        st.dataframe(corr_matrix)

        st.markdown("""
        This is a correlation matrix for all the numeric features in our combined wine DataFrame. Each cell shows the Pearson correlation coefficient between two variables, ranging from –1 (perfect inverse linear relationship) to +1 (perfect direct linear relationship).

        Examining this data for key takeaways we can safely assume that alcohol and quality columns have the strongest correlation in the dataset. Meaning the higher the alcohol content of the wine the higher its quality is usually, - which is something everyone can already agree with.

        When looking for the lowest influence on the quality of wine, we look for the most negative number which is density. Density then least affects the quality of the wine.
        """)

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

        # === Residual Sugar by Wine Type ===
        st.subheader("Residual Sugar by Wine Type")
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        sns.boxplot(x='type', y='residual sugar', data=both_wine, ax=ax4)
        ax4.set_title('Residual Sugar by Wine Type')
        st.pyplot(fig4)

        st.write("""
        White wine has significantly more residual sugar but as seen in our correlation matrix residual sugar does not impact the quality of wine significantly. It's pearson correlation being 0.036980, so very little to no impact. We can check this further using a scatterplot:
        """)

        # === Residual Sugar vs. Quality ===
        st.subheader("Residual Sugar vs. Quality")
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x='residual sugar', y='quality', hue='type', data=both_wine, alpha=0.5, ax=ax5)
        ax5.set_title('Residual Sugar vs. Quality')
        st.pyplot(fig5)

        st.write("""
        We can see in the scatterplot, that there are some outliers, especially in the white wine entries. Values above 25 g/L are rare and far from the data cluster.
        """)

        # === Outliers ===
        Q1 = both_wine['residual sugar'].quantile(0.25)
        Q3 = both_wine['residual sugar'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find outlier rows
        outliers = both_wine[(both_wine['residual sugar'] < lower_bound) | (both_wine['residual sugar'] > upper_bound)]

        # Display the rows with outliers
        st.subheader("Outlier rows (based on residual sugar):")
        st.dataframe(outliers[['residual sugar', 'type']])

        st.subheader("Index positions of outlier rows:")
        st.write(outliers.index.tolist())

        st.subheader("Number of outliers:")
        st.write(len(outliers))

        st.write("""
        As we can see, there are 118 outliers in both wines. I have here found the outliers in the 0.25 & 0.75 quantiles to outliers on both sides. We will now remove them.
        """)

        # === Remove Outliers ===

        both_wine_no_outliers = both_wine.drop(outliers.index).reset_index(drop=True)

        st.write("""
        After removing the outliers we'll first check to see if it worked. First we look at how many outliers have been removed. Then we check, if the extreme values are still there, and finally we make two boxplots to see the difference.
        """)

        st.subheader("Outlier Removal Summary")
        st.write("Original number of rows:", len(both_wine))
        st.write("Number of rows after outlier removal:", len(both_wine_no_outliers))
        st.write("Number of rows removed:", len(both_wine) - len(both_wine_no_outliers))

        st.subheader("Maximum Residual Sugar Comparison")
        st.write("Max residual sugar before:", both_wine['residual sugar'].max())
        st.write("Max residual sugar after:", both_wine_no_outliers['residual sugar'].max())

        # === Residual Sugar by Wine Type (Before Removal) ===
        st.subheader("Residual Sugar by Wine Type (Before Removal)")
        fig6, ax6 = plt.subplots(figsize=(6, 5))
        sns.boxplot(x='type', y='residual sugar', data=both_wine, ax=ax6)
        ax6.set_title('Residual Sugar by Wine Type (Before Removal)')
        st.pyplot(fig6)

        # === Residual Sugar by Wine Type (After Removal) ===
        st.subheader("Residual Sugar by Wine Type (After Removal)")
        fig7, ax7 = plt.subplots(figsize=(6, 5))
        sns.boxplot(x='type', y='residual sugar', data=both_wine_no_outliers, ax=ax7)
        ax7.set_title('Residual Sugar by Wine Type (After Removal)')
        st.pyplot(fig7)

        # === Exploring the Relationship Between pH and Density Using Binning ===
        st.subheader("Exploring the Relationship Between pH and Density Using Binning")
        st.write("""
        Binning data essentially means slicing the data into short bits. Slicing the cake into pieces and looking at each piece individually. When we slice pH into 5 and 10 bins, we in the following code examine which slice has the heavist wines (highest average density) 
        """)

        # === 5 Bins ===
        ph_bins_5 = pd.cut(both_wine['pH'], bins=5)
        density_by_ph_5 = both_wine.groupby(ph_bins_5, observed=True)['density'].mean()

        st.subheader("Density per pH Bin (5 bins)")
        st.dataframe(density_by_ph_5.reset_index().rename(columns={'density': 'Average Density'}))

        st.write(
            f"Highest density (5 bins): {density_by_ph_5.idxmax()} with value {density_by_ph_5.max():.4f}"
        )

        # === 10 Bins ===
        ph_bins_10 = pd.cut(both_wine['pH'], bins=10)
        density_by_ph_10 = both_wine.groupby(ph_bins_10, observed=True)['density'].mean()

        st.subheader("Density per pH Bin (10 bins)")
        st.dataframe(density_by_ph_10.reset_index().rename(columns={'density': 'Average Density'}))

        st.write(
            f"Highest density (10 bins): {density_by_ph_10.idxmax()} with value {density_by_ph_10.max():.4f}"
        )

        st.write("""
        Using 5 bins we notice that the highest avg. density is in the "3.494 - 3.752" with a value of 0.994.

        Using 10 bins means we have more data to work with (slicing the cake into more pieces) and we see that the pH range "3.365 - 3.494" has the highest avg. density. 

        5 bins gave a broad peak (3.494–3.752), but 10 bins pinpointed the true maximum/optimal pH range for density (3.365–3.494).

        What does this actually mean? 

        It means that wine density peaks at a mid-range pH (3.365–3.494), not at the highest pH.
        """)

        st.subheader("Correlation between pH and Density")
        st.dataframe(both_wine[['pH', 'density']].corr())

        st.subheader("Correlation between Density and Quality")
        st.dataframe(both_wine[['density', 'quality']].corr())

        st.subheader("Correlation between Density and Alcohol")
        st.dataframe(both_wine[['density', 'alcohol']].corr())

        # === Correlation Matrix ===
        corr_matrix = both_wine.drop(columns='type_code').corr(numeric_only=True)

        fig9, ax9 = plt.subplots(figsize=(10, 8))
        im = ax9.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        # Add color bar
        plt.colorbar(im, ax=ax9)

        # Add ticks and labels
        ax9.set_xticks(np.arange(len(corr_matrix.columns)))
        ax9.set_yticks(np.arange(len(corr_matrix.columns)))
        ax9.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax9.set_yticklabels(corr_matrix.columns)

        # Add values to each cell
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                ax9.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                        ha='center', va='center', color='black', fontsize=8)

        ax9.set_title('Correlation Matrix (Matplotlib)', fontsize=14)
        plt.tight_layout()

        st.subheader("Correlation Matrix (Matplotlib)")
        st.pyplot(fig9)

        # 1. View correlation matrix
        corr_matrix = both_wine.corr(numeric_only=True)

        # 2. Check correlation with 'quality'
        quality_corr = corr_matrix['quality'].drop('quality')
        low_corr = quality_corr[(quality_corr.abs() < 0.1) & (quality_corr.abs() > 0)]

        st.subheader("Correlation Matrix")
        st.dataframe(corr_matrix)

        st.subheader("Variables Weakly Correlated with Quality (|correlation| < 0.1)")
        st.dataframe(low_corr.reset_index().rename(columns={'index': 'Variable', 'quality': 'Correlation with Quality'}))

        st.write("""
        In the figure above we have identified features that has a weak correlation to quality.
        """)

        # === Data After Removing Weakly Correlated Features ===
        weak_features = [
            'fixed acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide',
            'total sulfur dioxide', 'pH', 'sulphates'
        ]

        both_wine_reduced = both_wine.drop(columns=weak_features)

        st.subheader("Data After Removing Weakly Correlated Features")
        st.write(f"Removed features: {', '.join(weak_features)}")
        st.dataframe(both_wine_reduced)

        # === PCA Visualization of Wine Dataset After Feature Reduction ===

        # Step 1: Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(both_wine_reduced.select_dtypes(include='number'))

        # Step 2: Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        # Step 3: Add PCA results to the DataFrame
        both_wine_reduced['PCA1'] = pca_result[:, 0]
        both_wine_reduced['PCA2'] = pca_result[:, 1]

        # Step 4: Explained variance ratio
        st.subheader("Explained Variance Ratio by PCA Components")
        st.write(f"PCA1: {pca.explained_variance_ratio_[0]:.2%}")
        st.write(f"PCA2: {pca.explained_variance_ratio_[1]:.2%}")

        # Step 5: Visualize PCA results
        fig10, ax10 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='PCA1', y='PCA2', hue='type', data=both_wine_reduced, alpha=0.7, ax=ax10)
        ax10.set_title('PCA of Wine Dataset')
        st.pyplot(fig10)

        st.write("""
        What the PCA scatter plot shows us is essentially that there is a chemical difference between white and red wines. The wine samples have been put into two magic axes PC1 and PC2 that capture the biggest chunks of variation in the data and capture all the data together.

        PC1 explains about 44% of the difference between wines, and PC2 another 24% as seen in our explained variance ratio above the graph. We then see that red and white wine types form two disctinct clusters. Meaning that there actually is an obvious difference between red and white based on our data/columns.

        We do a "sanity check" by printing 10 random rows, just to see that nothing is wrong.
        """)

        random_rows = both_wine_reduced.sample(n=10)

        st.subheader("10 Random Rows from Reduced Wine Data")
        st.dataframe(random_rows)

        return both_wine, numeric_cols

    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
