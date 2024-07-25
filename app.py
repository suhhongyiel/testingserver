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

    # # 진행 바 초기화
    # progress_bar = st.progress(0)
    # progress_text = st.empty()
    # progress_step = 0

    # patients demographic 정보 기입
    patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=62)  # Default age is 62
    patient_sex = st.sidebar.selectbox("Sex", options=["Male", "Female", "Other"])
    cancer_type = st.sidebar.text_input("Cancer Type", value="Lung cancer")
    treatment_type = st.sidebar.text_input("Treatment Type", value="Chemotherapy + immunotherapy")

    try:
        
        query = "SELECT study_ID FROM fitbit_device_list"
        device_info_options = [row[0] for row in pd.read_sql(query, engine).values]
        device_info = st.sidebar.selectbox("Fitbit ID:", device_info_options)
        
        if device_info is None:
            st.warning("Please select a device ID.")
            return  # 디바이스 ID가 선택되지 않았다면 여기서 중단

        table_resting_heart_rate = f'{device_info}_휴식기심박수'
        df = pd.read_sql(f"SELECT date FROM {table_resting_heart_rate}", engine)
        df['date'] = pd.to_datetime(df['date'], format='mixed')

        st.write(df)

        if df['date'].isna().any():
            st.error("Some dates could not be converted and are missing (NaT). Please check the data.")
            return

        min_date = df['date'].min()
        max_date = df['date'].max()
        st.write("Earliest date in the data:", min_date)
        st.write("Latest date in the data:", max_date)

        max_date = max_date.to_pydatetime()
        min_date = min_date.to_pydatetime()

        if pd.notnull(min_date) and pd.notnull(max_date):
            start_date, last_date = st.slider("날짜 범위 선택:", 
                                            min_value=min_date, 
                                            max_value=max_date, 
                                            value=(min_date, max_date),
                                            format='YYYY-MM-DD')
        else:
            st.error("Valid date range is not available in the data.")
            return  # 유효한 날짜 범위가 없다면 여기서 중단
    except SQLAlchemyError as e:
        st.error(f"An error occurred with the database: {e}")
        return  # 데이터베이스 오류 발생 시 중단
    except ValueError as e:
        st.error(f"Date format error: {e}")
        return  # 날짜 형식 오류 발생 시 중단
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return  # 기타 예외 발생 시 중단

    # progress_step += 1
    # progress_bar.progress(min(progress_step / 10, 1.0))  # 1.0을 초과하지 않도록 함
    # progress_text.text("Loading heart rate data...")

    table_resting_heart_rate = f'{device_info}_휴식기심박수'
    table_heart_rate = f'{device_info}_분별심박수'
    table_activity = f'{device_info}_활동량'
    table_sleep_detail = f'{device_info}_수면상세'

    try:
        # SQL 형태의 데이터 호출 및 필터링을 쿼리에서 바로 수행
        resting_heart_rate_query = text(f"""
            SELECT * FROM {table_resting_heart_rate} 
            WHERE date BETWEEN :start_date AND :end_date
        """)
        heart_rate_query = text(f"""
            SELECT * FROM {table_heart_rate} 
            WHERE date BETWEEN :start_date AND :end_date
        """)
        activity_query = text(f"""
            SELECT * FROM {table_activity} 
            WHERE date BETWEEN :start_date AND :end_date
        """)
        sleep_detail_query = text(f"""
            SELECT * FROM {table_sleep_detail} 
            WHERE date BETWEEN :start_date AND :end_date
        """)
        resting_heart_rate_df = pd.read_sql(resting_heart_rate_query, engine, params={"start_date": start_date, "end_date": last_date})
        heart_rate_df = pd.read_sql(heart_rate_query, engine, params={"start_date": start_date, "end_date": last_date})
        activity_df = pd.read_sql(activity_query, engine, params={"start_date": start_date, "end_date": last_date})
        sleep_detail_df = pd.read_sql(sleep_detail_query, engine, params={"start_date": start_date, "end_date": last_date})

        # date를 datetime 형식으로 변환
        for df in [resting_heart_rate_df, heart_rate_df, activity_df, sleep_detail_df]:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)  # 날짜 변환 실패한 행 제거

        last_date = last_date.replace(hour=23, minute=59, second=0)

        # 필터링된 데이터프레임
        df_resting = resting_heart_rate_df[(resting_heart_rate_df['date'] >= start_date) & (resting_heart_rate_df['date'] <= last_date)]
        df_heart = heart_rate_df[(heart_rate_df['date'] >= start_date) & (heart_rate_df['date'] <= last_date)]
        df_activity = activity_df[(activity_df['date'] >= start_date) & (activity_df['date'] <= last_date)]
        df_sleep_detail = sleep_detail_df[(sleep_detail_df['date'] >= start_date) & (sleep_detail_df['date'] <= last_date)]

        fig = plt.figure(figsize=(40, 20))  # Increase figure size
        gs = gridspec.GridSpec(6, 1, figure=fig, height_ratios=[1, 2, 2, 2, 2, 2])

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])
        ax3 = fig.add_subplot(gs[3])
        ax4 = fig.add_subplot(gs[4])
        ax5 = fig.add_subplot(gs[5])

        # demographic_area를 맨 위 축에 추가
        ax0 = function.demographic_area(ax0, start_date, last_date, device_info, patient_age, patient_sex, cancer_type, treatment_type)

        ax1 = function.plot_compliance(ax1, df_heart, start_date, last_date)
        ax2 = function.heart_rate_plot(ax2, df_heart, start_date, last_date)
        ax3 = function.plot_activity(ax3, df_activity, start_date, last_date)
        ax4, df_based = function.sleep_graph_ver(ax4, df_sleep_detail, df_heart, start_date, last_date)
        ax5 = function.sleep_table_area(ax5, df_based, start_date, last_date)

        plt.tight_layout()
        st.pyplot(fig)
        # progress_bar.empty()
        # progress_text.empty()

        if st.button("Export to PDF"):
            function.export_plots_to_pdf(fig)

    except SQLAlchemyError as e:
        st.error(f"An error occurred with the database: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
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

