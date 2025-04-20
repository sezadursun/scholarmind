# 🎨 ScholarMind UI Enhancements (for streamlit_app.py)

import streamlit as st

st.set_page_config(
    page_title="ScholarMind – 📚 Akademik Hafıza Asistanı",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: nowrap;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
        gap: 0.4rem;
    }

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #F0F0F0;
        color: #333;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        border: 1px solid #d0d0d0;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        white-space: nowrap;
        flex-shrink: 0;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4B3F72 !important;
        color: white !important;
        border: 2px solid #4B3F72;
        box-shadow: inset 0 -4px 0 #F44336;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E0E0E0;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🧠 ScholarMind")
st.caption("Bilge araştırma hafızanız. Arayın, özetleyin, hatırlayın.")
