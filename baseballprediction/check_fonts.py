import matplotlib.font_manager as fm
import streamlit as st

font_list = sorted(set([fm.FontProperties(fname=f).get_name() for f in fm.findSystemFonts()]))
st.write("利用可能なフォント一覧:", font_list)
