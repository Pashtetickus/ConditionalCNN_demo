import streamlit as st
from scripts.prepare_data import load_weights
from src.app import main


if 'downloaded' not in st.session_state:
    load_weights()
    st.session_state['downloaded'] = True

main()

