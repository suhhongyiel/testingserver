import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import sqlite3 as lite
from datetime import datetime as dt

st.write("testing?")


title1 = st.text_input('UID', 'smcfb.01.0099')
title2 = st.text_input('START', '2023-07-14')
title3 = st.text_input('ACESS_TOKEN', 'eyHDSDS ...')

# check box 로 바꿔서 실시간 데이터 베이스 전송 가능
title4 = st.text_input('status', 'o OR x')

st.write('UID: ', title1)
st.write('START: ', title2)
st.write('ACESS_TOKEN: ', title3)
st.write('satus: ', title4)

