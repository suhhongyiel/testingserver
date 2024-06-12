import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil import parser
import pandas as pd
import io
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages
# import MatplotlibReportGenerator as mrg
import matplotlib.dates as mdates
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import utils
import function
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.gridspec as gridspec

# 데이터베이스 연결 설정
db_url = 'mysql+pymysql://root:Korea2022!@119.67.109.156:3306/project_wd'
engine = create_engine(db_url)

# < === 페이지 레이아웃 === >
def page_about():
    st.title("DataBase")

    # patients demographic 정보 기입
    patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=62)  # Default age is 62
    patient_sex = st.sidebar.selectbox("Sex", options=["Male", "Female", "Other"])
    cancer_type = st.sidebar.text_input("Cancer Type", value="Lung cancer")
    treatment_type = st.sidebar.text_input("Treatment Type", value="Chemotherapy + immunotherapy")

    try:
        query = "SELECT study_ID FROM fitbit_device_list"
        device_info_options = [row[0] for row in pd.read_sql(query, engine).values]
        # 해당 device 의 명단리스트를 쭉 부르고 
        # 해당 리스트를 선택하면 선택할 수 있는 slider 가 보이고 해당 slider 를 조작하면 해당되는 plot 들이 pdf 로 출력
        device_info = st.sidebar.selectbox("Fitbit ID:", device_info_options)
        
        if device_info is None:
            st.warning("Please select a device ID.")
            return  # 디바이스 ID가 선택되지 않았다면 여기서 중단

        table_resting_heart_rate = f'{device_info}_휴식기심박수' # 기본 date 범위를 알아보기위해 휴식기 심박수 데이터를 불러옴
        df = pd.read_sql(f"SELECT * FROM {table_resting_heart_rate}", engine)
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')

        # NaT 값이 있는지 확인하고, 있을 경우 처리
        if df['date'].isna().any():
            st.error("Some dates could not be converted and are missing (NaT). Please check the data.")
            return

        min_date = df['date'].min()
        max_date = df['date'].max()
        st.write("Earliest date in the data:", min_date)
        st.write("Latest date in the data:", max_date)

        # format 을 맞추기 위한 pydatetime 포멧 변경
        max_date = max_date.to_pydatetime()
        min_date = min_date.to_pydatetime()


        # 모든 데이터를 호출 한뒤 plot 을 진행
        if pd.notnull(min_date) and pd.notnull(max_date):
            start_date, last_date = st.slider("날짜 범위 선택:", 
                                            min_value=min_date, 
                                            max_value=max_date, 
                                            value=(min_date, max_date),
                                            format='YYYY-MM-DD')
            
        else:
            st.error("Valid date range is not available in the data.")
    except SQLAlchemyError as e:
        st.error(f"An error occurred with the database: {e}")
    except ValueError as e:
        st.error(f"Date format error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

    # device_info 는 smcfb 아이디임
    # 모든 데이터를 불러오기
    table_resting_heart_rate = f'{device_info}_휴식기심박수'
    table_heart_rate = f'{device_info}_분별심박수'
    table_activity = f'{device_info}_활동량'
    table_sleep_detail = f'{device_info}_수면상세'

    st.write(start_date)
    st.write(last_date)


    resting_heart_rate_df = pd.read_sql(f"SELECT * FROM {table_resting_heart_rate}", engine)
    heart_rate_df = pd.read_sql(f"SELECT * FROM {table_heart_rate}", engine)
    activity_df = pd.read_sql(f"SELECT * FROM {table_activity}", engine)
    sleep_detail_df = pd.read_sql(f"SELECT * FROM {table_sleep_detail}", engine)

    # date 가 -1 이면 해당 값은 drop 왜냐하면 해당 값은 말이 안되는 값이기 때문에
    resting_heart_rate_df = resting_heart_rate_df[resting_heart_rate_df['date'] != "-1"]
    heart_rate_df = heart_rate_df[heart_rate_df['date'] != "-1"]
    activity_df = activity_df[activity_df['date'] != "-1"]
    sleep_detail_df = sleep_detail_df[sleep_detail_df['date'] != "-1"]
    # date 가 -1.0 이면 해당 값은 drop: 이상치 처리
    resting_heart_rate_df = resting_heart_rate_df[resting_heart_rate_df['date'] != '-1.0']
    heart_rate_df = heart_rate_df[heart_rate_df['date'] != '-1.0']
    activity_df = activity_df[activity_df['date'] != '-1.0']
    sleep_detail_df = sleep_detail_df[sleep_detail_df['date'] != '-1.0']
    # date 를 실제 date 형식의 값으로 변경
    resting_heart_rate_df['date'] = pd.to_datetime(resting_heart_rate_df['date'], format='mixed')
    heart_rate_df['date'] = pd.to_datetime(heart_rate_df['date'], format='mixed')
    activity_df['date'] = pd.to_datetime(activity_df['date'], format='mixed')
    sleep_detail_df['date'] = pd.to_datetime(sleep_detail_df['date'], format='mixed')

    # date 를 23:59:59 형태로 변경해야함
    last_date = last_date.replace(hour=23, minute=59, second=0)

    st.write(last_date)



    # 해당 데이터 프레임을 실제 입력된 date 로 변경
    df_resting = resting_heart_rate_df[(resting_heart_rate_df['date'] >= start_date) & (resting_heart_rate_df['date'] <= last_date)]
    df_heart = heart_rate_df[(heart_rate_df['date'] >= start_date) & (heart_rate_df['date'] <= last_date)]
    df_activity = activity_df[(activity_df['date'] >= start_date) & (activity_df['date'] <= last_date)]
    df_sleep_detail = sleep_detail_df[(sleep_detail_df['date'] >= start_date) & (sleep_detail_df['date'] <= last_date)]


    # demographic 설정 항목 진행

    # ax0 = plt.axes((-0.06, 0.85, 0.9, 0.12)) # demographic
    # ax1 = plt.axes((0.06, 0.68, 0.9, 0.14)) # Compilance
    # ax2 = plt.axes((0.06, 0.50, 0.9, 0.14)) # Physical activity
    # ax3 = plt.axes((0.06, 0.32, 0.9, 0.14)) # Daily Heart Rate
    # ax4 = plt.axes((0.06, 0.152, 0.9, 0.14)) # Sleep Detail
    # ax5 = plt.axes((0.06, 0.01, 0.9, 0.14)) # Sleep States
    

    # # 그래프 설정하기
    # fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, 1, figsize=(10, 15))  # 2개의 축을 생성
    
    # fig.subplots_adjust(hspace=0)

    fig = plt.figure(figsize=(10, 15))
    gs = gridspec.GridSpec(6, 1, figure=fig, height_ratios=[2, 2, 2, 2, 2, 1])

    # 각 축 할당
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    ax4 = fig.add_subplot(gs[4])
    ax5 = fig.add_subplot(gs[5])

    
    # ax function 에서 불러오기
    # ax0 = function.demographic_area(ax0, start_date, last_date, device_info)
    ax0 = function.demographic_area(ax0, start_date, last_date, device_info, patient_age, patient_sex, cancer_type, treatment_type)
    ax1 = function.plot_compliance(ax1, df_heart, start_date, last_date)
    ax2 = function.heart_rate_plot(ax2, df_heart, start_date, last_date)
    ax3 = function.plot_activity(ax3, df_activity, start_date, last_date)
    ax4, df_based = function.sleep_graph_ver(ax4, df_sleep_detail, df_heart, start_date, last_date)
    # df_based 는 sleep 라벨을 sleep, wake, missing compliance 를 labeling 한 것 들 임
    ax5 = function.sleep_table_area(ax5, df_based, start_date, last_date)

    plt.tight_layout()
    st.pyplot(fig)
    # 해당 info 를 불러온 후 dbeaver 가 알아 먹는 parameter 로 replace 하여 해당 table 호출
    # i.e. smcfb.01.001_분별심박수, smcfb.01.001_활동량 .. etc

    # Add a button to trigger PDF export
    st.write("Ready to export PDF")
    if st.button("Export to PDF"):
        function.export_plots_to_pdf(fig)
    
# Usage example
def main():
    pages = {
        "환자데이터베이스": page_about
    }
    selected_page = "환자데이터베이스"
    if selected_page in pages:
        pages[selected_page]()

if __name__ == '__main__':
    main()



# ======================================================================

