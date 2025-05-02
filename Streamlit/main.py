import streamlit as st
import WhiteWine

def show_homepage():
    st.title('Mini Project 2 - Wine data')
    st.write('Group 6')

    st.write('Made by: Bekhan, Otto, Victor & Patrick')
    st.write("""
    ### 🍇 Understanding the Complexity of Wine Quality

    Wine quality is influenced by a myriad of factors that extend beyond measurable chemical properties. While our dataset provides insights into attributes like acidity, sugar content, and alcohol levels, it's essential to recognize the significant roles played by geographical origin, production techniques, and human perception.

    **Geographical Origin (Terroir):**  
    The term *terroir* encapsulates the unique combination of soil, climate, topography, and other environmental factors of a vineyard. These elements profoundly affect grape characteristics, leading to distinct flavors and aromas in the resulting wine. For instance, variations in temperature and rainfall can influence grape ripeness and acidity levels, thereby impacting taste profiles.

    **Production Techniques:**  
    Winemaking methods, including fermentation processes, aging, and the choice of yeast strains, significantly contribute to a wine's quality. Techniques such as oak barrel aging can impart additional flavors and complexity, while decisions made during fermentation can affect the wine's clarity, stability, and aromatic profile.

    **Human Perception:**  
    Ultimately, wine quality is also a subjective experience. Individual preferences, cultural backgrounds, and sensory perceptions play crucial roles in how a wine is evaluated and enjoyed. Factors like aroma, taste, and mouthfeel are interpreted differently by each person, making wine appreciation a deeply personal journey.

    By integrating data analysis with an understanding of these qualitative aspects, we aim to provide a holistic view of what defines wine quality.""")


def main():
    st.sidebar.title("Wine Data")
    page = st.sidebar.selectbox("Choose a page", ["Homepage","WhiteWine"])

    if page == "Homepage":
        show_homepage()
    elif page == "WhiteWine":
        WhiteWine.WhiteWine()


if __name__ == "__main__":
    main()
