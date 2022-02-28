import glob
import os
from pathlib import Path

import streamlit as st
from scripts.prepare_data import load_weights
from src.app import main

if 'downloaded' not in st.session_state:
    load_weights()
    st.session_state['downloaded'] = True

    for gdown_tmp_dir in glob.glob(str(Path.home()) + '/.cache/gdown/*tmp*'):
        os.rmdir(gdown_tmp_dir)

main()

