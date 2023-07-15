import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime as dt
import pymysql

st.write("testing?")


UID = st.text_input('UID', 'smcfb.01.0099')
START = st.text_input('START', '2023-07-14')
ACCESS_TOKEN = st.text_input('ACESS_TOKEN', 'eyHDSDS ...')

# check box 로 바꿔서 실시간 데이터 베이스 전송 가능
status = st.text_input('status', 'o OR x')

st.write('UID: ', UID)
st.write('START: ', START)
st.write('ACESS_TOKEN: ', ACCESS_TOKEN)
st.write('satus: ', status)


if 'submitted' not in st.session_state:
    st.session_state.submitted = False

def update():
    st.session_state.submitted = True

# Your Form Here
st.form_submit_button('Submit', on_click=update)

if st.session_state.submitted:
    st.write('Form submitted')

# # DB input and connect
# db = pymysql.connect(host='119.67.109.156', 
#                 port=3306,
#                 user='root', 
#                 password='Korea2022!', 
#                 db='project_wd', 
#                 charset='utf8')
# cursor = db.cursor()

# sql2 = "INSERT INTO device_info (UID, START, ACCESS_TOKEN, status) VALUES (%s, %s, %s, %s)"
# cursor.execute(sql2, ("smcfb.01.0099", "2023-07-14", "ee44", "o"))
# db.commit()
# db.close()