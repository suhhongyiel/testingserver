import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime as dt
import pymysql

st.write("testing?")







def on_button_click():
    st.session_state.error_message = ''
    st.session_state.result_message = ''
    if not str(st.session_state.UID) and not str(st.session_state.START) and not str(st.session_state.ACESS_TOKEN) and not str(st.session_state.status):
        st.session_state.error_message = "포멧을 정확히 입력해주세요"
    else:
        st.session_state.result_message = f"DB was updated {str(st.session_state.UID)}"


st.title("Streamlit Test")

input_user_name = st.text_input(key = 'UID', value = '')
input_start_date = st.text_input(key = 'START', value = '2023-07-14')
input_access_token = st.text_input(key = 'ACESS_TOKEN', value = 'eyHDSDS ...')
# check box 로 바꿔서 실시간 데이터 베이스 전송 가능
status = st.text_input('status', 'o OR x')

st.write('UID: ', input_user_name)
st.write('START: ', input_start_date)
st.write('ACESS_TOKEN: ', input_access_token)
st.write('satus: ', status)

checkbox = st.checkbox('agree')
st.button("Confirm", key='confirm_btn', disabled=(checkbox is False), on_click=on_button_click)

con = st.container()
con.caption("Result")
if 'error_message' in st.session_state and st.session_state.error_message:
    con.error(st.session_state.error_message)
if 'result_message' in st.session_state and st.session_state.result_message:
    con.write(st.session_state.result_message)

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