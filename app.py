import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime as dt
import pymysql

st.write("testing?")
db = pymysql.connect(host='119.67.109.156', 
                        port=3306,
                        user='root', 
                        password='Korea2022!', 
                        db='project_wd', 
                        charset='utf8')




def page_home():
    # Confirm 버튼 클릭시에 발생
    def on_button_click():
        st.session_state.error_message = ''
        st.session_state.result_message = ''

        # 포멧 정하기
        if 'smcfb' not in st.session_state.UID:
            st.session_state.error_message = "UID 포멧을 정확히 입력해주세요"
        elif not str(st.session_state.START):
            st.session_state.error_message = "시작 날짜 포멧을 정확히 입력해주세요"
        elif not len(str(st.session_state.ACESS_TOKEN)) == 273:
            st.session_state.error_message = "Acess Token 포멧을 정확히 입력해주세요"
        elif str(st.session_state.status) != 'o' and str(st.session_state.status) != 'x':
            st.session_state.error_message = "Status 포멧을 정확히 입력해주세요 (o/x)"
        
        else:
            st.session_state.result_message = f"DB was updated {str(st.session_state.UID)}"


    st.title("Streamlit Test")

    input_user_name = st.text_input(label="User Name", key = 'UID', value = '')
    input_start_date = st.text_input(label="start",key = 'START', value = '2023-07-14')
    input_access_token = st.text_input(label="acess_toiken",key = 'ACESS_TOKEN', value = 'eyHDSDS ...')
    # check box 로 바꿔서 실시간 데이터 베이스 전송 가능
    status = st.text_input(label="status", key = 'status', value = 'o OR x')

    # 입력된 데이터 출력
    st.write('UID: ', input_user_name)
    st.write('START: ', input_start_date)
    st.write('ACESS_TOKEN: ', input_access_token)
    st.write('status: ', status)


    # 체크 박스를 통해 시그널 전송 (상태 전송)
    checkbox = st.checkbox('버튼잠금풀기')
    confirm_btn = st.button("Confirm", key='confirm_btn', disabled=(checkbox is False), on_click=on_button_click)

    con = st.container()
    con.caption("Result")
    signal = 0
    if 'error_message' in st.session_state and st.session_state.error_message:
        con.error(st.session_state.error_message)
        signal = 0
    if 'result_message' in st.session_state and st.session_state.result_message:
        con.write(st.session_state.result_message)
        st.write("Click button")
        signal = 1

    if signal == 1:
        # DB input and connect
        
        cursor = db.cursor()

        sql2 = "INSERT IGNORE INTO device_info (UID, START, ACCESS_TOKEN, status) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql2, (input_user_name, input_start_date, input_access_token, status))
        db.commit()
        db.close()


# 두번째 페이지
def page_about():
    st.title("About Page")
    # Add content for the about page
    with db.cursor() as cursor:
        cursor.execute("SELECT UID FROM device_info")
        device_info_options = [row[0] for row in cursor.fetchall()]

    device_info = st.selectbox("SELECT Device Info", device_info_options)
    table_name = f"{device_info}"
    st.info(f"Selected Device Info: {device_info}")
    st.info(f"Table Name: {table_name}")
    all_name = get_table_names(table_name)
    

    smcfb_info = st.selectbox("SELECT SMCFB Info", all_name)


# dbeaver 에서 해당되는 데이터 테이블 가져오기
def get_table_names(table_name):
    table_names = []
    try:
        with db.cursor() as cursor:

            query = "SHOW TABLES LIKE %s"
            cursor.execute(query, (f"%{table_name}%",))
            result = cursor.fetchall()
            for row in result:
                table_names.append(row[0])



    except pymysql.Error as e:
        print(f"An error occurred: {e}")

    return table_names


# Usage example




















def main():
    pages = {
        "Home": page_home,
        "About": page_about,
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Execute the selected page function
    pages[selection]()

if __name__ == '__main__':
    main()