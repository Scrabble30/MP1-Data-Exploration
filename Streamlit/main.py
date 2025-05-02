import streamlit as st
import WhiteWine

def show_homepage():
    st.title('Mini Project 2 - Wine data')
    st.write('Group 6')

    st.write('Made by: Bekhan, Otto, Victor & Patrick')

def main():
    st.sidebar.title("Wine Data")
    page = st.sidebar.selectbox("Choose a page", ["Homepage","WhiteWine"])

    if page == "Homepage":
        show_homepage()
    elif page == "WhiteWine":
        WhiteWine.WhiteWine()


if __name__ == "__main__":
    main()
