# ScholarMind – UI Theme Configuration

import streamlit as st

def apply_scholarmind_theme():
    st.set_page_config(
        page_title="ScholarMind – 🧠 Akademik Hafıza Asistanı",
        page_icon="🧠",
        layout="wide"
    )
    st.markdown("""
        <style>
            body {
                background-color: #FAF9F6;
                font-family: 'Inter', sans-serif;
            }
            .block-container {
                padding-top: 2rem;
            }
            h1 {
                color: #4B3F72;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #C8CCD4;
                color: black;
            }
            .stTabs [aria-selected="true"] {
                background-color: #4B3F72;
                color: white;
            }
            .stButton>button {
                background-color: #3766E8;
                color: white;
                border-radius: 5px;
            }
        </style>
    """, unsafe_allow_html=True)
