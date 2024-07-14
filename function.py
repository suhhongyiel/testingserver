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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors

# csv 에서 drop 하는 컬럼 조건이 다르면 안됨
# 해당 컬럼에서 조건이 다르면 해당 조건에서 0과 -1에서 drop 되는 날짜의 갯수들이 다르기 때문
# 시간 데이터 변환 함수 정의
def normalize_time(time_str):
    try:
        # 시간 형식에 AM/PM이 포함된 경우
        if 'AM' in time_str or 'PM' in time_str:
            return pd.to_datetime(time_str, format='%I:%M:%S %p').strftime('%H:%M:%S')
        
        # ':'의 개수로 형식 판단
        parts = time_str.split(':')
        if len(parts) == 2:  # mm:ss.0 형식
            minutes, seconds = parts
            seconds = seconds.split('.')[0]
            return f"00:{minutes}:{seconds}"
        elif len(parts) == 3:  # hh:mm:ss.0 형식
            hours, minutes, seconds = parts
            seconds = seconds.split('.')[0]
            return f"{hours}:{minutes}:{seconds}"
        
    except Exception as e:
        print(f"Error converting time: {e}")
        return None

# 데이터베이스 연결 설정
db_url = 'mysql+pymysql://root:Korea2022!@119.67.109.156:3306/project_wd'
engine = create_engine(db_url)

def heart_rate_plot(ax, df, start_date, end_date):
    try:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        daily_mean = df.groupby(df['date'].dt.date)['value'].mean()

        # DataFrame으로 변환하여 필터링 가능하게 조정
        daily_mean = daily_mean.reset_index()
        daily_mean.columns = ['date', 'value']
        daily_mean['date'] = pd.to_datetime(daily_mean['date'])

        df_filtered = daily_mean[(daily_mean['date'] >= start_date) & (daily_mean['date'] <= end_date)]

        quartile_data = df_filtered.groupby(df_filtered['date'].dt.date)['value'].quantile([0.25, 0.75]).unstack()

        window_size = 3
        smoothed_mean = df_filtered['value'].rolling(window=window_size, center=True).mean()
        smoothed_q1 = quartile_data[0.25].rolling(window=window_size, center=True).mean()
        smoothed_q3 = quartile_data[0.75].rolling(window=window_size, center=True).mean()

        ax.plot(df_filtered['date'], smoothed_mean, label='Mean', color='magenta')
        ax.fill_between(df_filtered['date'], smoothed_q1, smoothed_q3, color='gray', alpha=0.2,
                        label='Q1-Q3 (Interquartile Range)')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.set_xlim([start_date, end_date])

        ax.set_xlabel('Date')
        ax.set_ylabel('Average Heart Rate')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_visible(False)
    except Exception as e:
        print(f"An error occurred: {e}")

    return ax

def plot_activity(ax, df, start_date, end_date):
    try:
        df['date'] = pd.to_datetime(df['date'])
        df = df.groupby('date', as_index=False)['steps'].sum()
        df.sort_values('date', inplace=True)
        
        df['steps'] = pd.to_numeric(df['steps'], errors='coerce')
        df = df.dropna(subset=['steps'])
        
        time = df['date']
        steps = df['steps']

        

        total_steps = steps.sum()
        daily_mean_steps = steps.mean()

        ax.plot(time, steps, '-o', label="Steps", color='navy')
        ax.set_ylabel('Steps')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels([])

        for date in time:
            ax.axvline(x=date, color='grey', linestyle='--', alpha=0.5)

        df['week'] = df['date'].dt.isocalendar().week
        weekly_mean_steps = df.groupby('week')['steps'].mean()
        weekly_mean = weekly_mean_steps.mean()
        ax.xaxis.set_visible(False)
        return ax

    except Exception as e:
        print(f"An error occurred: {e}")
        return

def sleep_graph_ver(ax, slp_df, hrdf, start_date, end_date):
    try:
        # 시간 데이터 변환
        slp_df['converted_time'] = slp_df['time_stamp'].apply(normalize_time)
        slp_df['datetime'] = pd.to_datetime(slp_df['date'].dt.date.astype(str) + ' ' + slp_df['converted_time'], errors='coerce')
        hrdf['converted_time'] = hrdf['time_min'].apply(normalize_time)
        hrdf['datetime'] = pd.to_datetime(hrdf['date'].dt.date.astype(str) + ' ' + hrdf['converted_time'], errors='coerce')
        
        # start time 세팅
        date_range = pd.date_range(start=start_date, end=end_date, freq='T')  # 'T'는 분(minute) 단위를 의미합니다.
        
        df_based = pd.DataFrame(date_range, columns=['datetime'])

        # hr setting
        hrdf_datetimes = set(hrdf['datetime'])
        df_based['value'] = df_based['datetime'].apply(lambda x: 2 if x in hrdf_datetimes else 1)


        # 데이터 확장
        expanded_rows = []
        for _, row in slp_df.iterrows():
            start_datetime = row['datetime']
            duration_seconds = int(row['sleep_duration'])  # 'duration'은 초 단위로 주어짐
            end_datetime = start_datetime + pd.Timedelta(seconds=duration_seconds)

            # 시작 시간부터 종료 시간까지 모든 분 생성
            current_datetime = start_datetime
            while current_datetime < end_datetime:
                expanded_rows.append({
                    'datetime': current_datetime,
                    'sleep_stage': row['sleep_stage']
                })
                current_datetime += pd.Timedelta(minutes=1)

        # 확장된 데이터 프레임 생성
        expanded_df = pd.DataFrame(expanded_rows)
        expand_datetimes = set(expanded_df['datetime'])
        df_based['events'] = df_based['datetime'].apply(lambda x: 0 if x in expand_datetimes else None)
        df_based['value'] = df_based['events'].combine_first(df_based['value'])
        df_based.drop('events', axis=1, inplace=True)

        # 'datetime' 열을 datetime 객체로 변환
        df_based['datetime'] = pd.to_datetime(df_based['datetime'])

        # 데이터를 날짜와 시간으로 분리
        df_based['date'] = df_based['datetime'].dt.date
        df_based['hour'] = df_based['datetime'].dt.hour

        # 유니크한 날짜를 정렬하여 사용
        unique_dates = df_based['date'].unique()
        unique_dates.sort()

        ax.set_xticklabels([])
        # 각 날짜와 시간대별로 값 플롯
        for date in unique_dates:
            day_data = df_based[df_based['date'] == date]
            for hour in range(24):
                hour_data = day_data[day_data['hour'] == hour]
                if not hour_data.empty:
                    value = int(hour_data.iloc[0]['value'])  # 첫 번째 값을 사용하고 정수형으로 변환
                    bars = ax.bar(date, 1, bottom=hour, color=['lightcoral', 'lightgray', 'lightgreen'][value % 3], align='edge')
            
        

        # 그리드 설정
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(24, 0)
        ax.set_xlim(left=pd.to_datetime(start_date), right=pd.to_datetime(end_date))

        # 축 설정 후 그리드 활성화
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        ax.set_xlabel('Date')
        ax.set_ylabel('Hour of Day')
        ax.xaxis.set_visible(False)
        return ax, df_based

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

def plot_compliance(ax, df, start_date, end_date):
    try:
        
        df['datetime'] = pd.to_datetime(df['date'].dt.date.astype(str) + ' ' + df['time_min'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        
        df = df.drop_duplicates(subset=['user_id', 'datetime'], keep='first')
        
        df['valid'] = (df['value'] != -1) & (df['value'] != 0)

        daily_compliance = df.groupby(df['datetime'].dt.date)['valid'].sum()
        daily_total = df.groupby(df['datetime'].dt.date)['valid'].count()

        daily_compliance_rate = (daily_compliance / 1440) * 100
        daily_compliance_rate.plot(kind='bar', color='skyblue', ax=ax)

        ax.set_xlabel('Date')
        ax.set_ylabel('Compliance Rate (%)')
        ax.set_title('Daily Heart Rate Compliance')


        # ax 의 x 축 제거
        ax.xaxis.set_visible(False)

        return ax
    
    except Exception as e:
        print(f"Error during plot_compliance: {e}")
        return
    
def demographic_area(ax, start_date, end_date, id, age, sex, cancer_type, treatment_type):
    fontdict = {
        'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 16,
    }
    texts = [
        f"Fitbit ID: {id}",
        f"Tracking date: {start_date.strftime('%Y-%m-%d')} (Start) - {end_date.strftime('%Y-%m-%d')} (End)",
        f"Patient info: Age {age}Y, Sex: {sex}",
        f"Cancer type: {cancer_type}",
        f"Treatment Type: {treatment_type}",
        f"Data extraction date: {datetime.today().strftime('%Y-%m-%d')}"
    ]
    # Add text line by line
    for i, text in enumerate(texts):
        ax.text(-0.1, 1 - 0.2 * i, text, transform=ax.transAxes, va='top', ha='left')

    ax.axis('off')  # Hide the axes
    return ax


def sleep_table_area(ax, df, start_date, end_date):
    try:
        # 데이터 로드 및 날짜 필터링
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))]

        # 날짜별 라벨 집계
        df['date'] = df['datetime'].dt.date
        unique_dates = sorted(df['date'].unique())
        date_index_map = {date: idx for idx, date in enumerate(unique_dates)}
        df['date_index'] = df['date'].map(date_index_map)

        df['label'] = df['value'].map({0: 'sleep', 1: 'missing', 2: 'wake'})
        
        # 라벨별 갯수 / 60 으로 해당 값을 H 로 치환
        daily_counts = df.groupby(['date', 'label']).size().unstack(fill_value=0)
        daily_counts = (daily_counts / 60).round(1)
        pivot_table = daily_counts.T

        # 날짜를 5일 간격으로 병합
        merged_dates = [unique_dates[i] for i in range(0, len(unique_dates), 5)]
        merged_pivot_table = pivot_table.groupby(lambda x: x // 5, axis=1).sum()
        merged_pivot_table.columns = merged_dates

        # 피벗 테이블을 텍스트 테이블로 플롯
        table = ax.table(cellText=merged_pivot_table.values, colLabels=merged_pivot_table.columns, rowLabels=merged_pivot_table.index, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)  # 폰트 크기 설정
        table.scale(1.0, 1.5)  # 표 크기 조정 (너비, 높이)
        
        # 색상 정의
        colors = {
            'sleep': mcolors.CSS4_COLORS['lightcoral'],
            'missing': mcolors.CSS4_COLORS['lightgrey'],
            'wake': mcolors.CSS4_COLORS['lightgreen']
        }

        # 행 색상 설정
        for (i, key) in enumerate(merged_pivot_table.index):
            for j in range(len(merged_pivot_table.columns)):
                table[(i+1, j)].set_facecolor(colors.get(key, 'white'))

        ax.axis('off')  # 축 비활성화

    except Exception as e:
        print(f"An error occurred: {e}")
    
    return ax
# def sleep_table_area(ax, df, start_date, end_date):
#     try:
#         # 데이터 로드 및 날짜 필터링
#         df['datetime'] = pd.to_datetime(df['datetime'])
        
#         # 날짜별 라벨 집계
#         df['date'] = df['datetime'].dt.date
#         unique_dates = sorted(df['date'].unique())
#         date_index_map = {date: idx for idx, date in enumerate(unique_dates)}
#         df['date_index'] = df['date'].map(date_index_map)

#         df['label'] = df['value'].map({0: 'sleep', 1: 'missing', 2: 'wake'})
        
#         # 라벨별 갯수 / 60 으로 해당 값을 H 로 치환
#         daily_counts = df.groupby(['date', 'label']).size().unstack(fill_value=0)
#         daily_counts = (daily_counts / 60).round(1)
#         pivot_table = daily_counts.T

#         # 피벗 테이블 생성: 라벨별, 날짜별 집계
#         # pivot_table = df.pivot_table(index='label', columns='date_index', aggfunc='size', fill_value=0)

#         # 피벗 테이블을 텍스트 테이블로 플롯
#         table = ax.table(cellText=pivot_table.values, colLabels=pivot_table.columns, rowLabels=pivot_table.index, loc='center')
#         table.auto_set_font_size(False)
#         table.set_fontsize(10)  # 폰트 크기 설정
#         table.scale(1.0, 1.5)  # 표 크기 조정 (너비, 높이)
        
#                 # 색상 정의
#         colors = {
#             'sleep': mcolors.CSS4_COLORS['lightcoral'],
#             'missing': mcolors.CSS4_COLORS['lightgrey'],
#             'wake': mcolors.CSS4_COLORS['lightgreen']
#         }

#         # 행 색상 설정
#         for (i, key) in enumerate(pivot_table.index):
#             for j in range(len(pivot_table.columns)):
#                 table[(i+1, j)].set_facecolor(colors.get(key, 'white'))


#         ax.axis('off')  # 축 비활성화

#     except Exception as e:
#         print(f"An error occurred: {e}")
    
#     return ax

# def sleep_table_area(ax, df, start_date, end_date):
#     try:
#         # 데이터 로드 및 날짜 필터링
#         df['datetime'] = pd.to_datetime(df['datetime'])
        
#         # 날짜 및 시간별 라벨 집계
#         df['hour'] = df['datetime'].dt.floor('H')  # 시간 단위로 바꿈
#         df['date'] = df['datetime'].dt.date

#         # 날짜 필터링
#         df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

#         # 시간별 집계
#         df['label'] = df['value'].map({0: 'sleep', 1: 'wake', 2: 'missing'})
#         pivot_table = df.pivot_table(index=['label'], columns=df['hour'].dt.strftime('%Y-%m-%d %H:%M'), aggfunc='size', fill_value=0)
        
#         # 피벗 테이블을 텍스트 테이블로 플롯
#         table = ax.table(cellText=pivot_table.values, colLabels=pivot_table.columns, rowLabels=pivot_table.index, loc='center')
#         table.auto_set_font_size(False)
#         table.set_fontsize(10)  # 폰트 크기 설정
#         table.scale(1.0, 1.5)  # 표 크기 조정 (너비, 높이)

#         ax.axis('off')  # 축 비활성화

#     except Exception as e:
#         print(f"An error occurred: {e}")
    
#     return ax

# PDF export function
def export_plots_to_pdf(fig, filename='report.pdf'):
    with PdfPages(filename) as pdf:
        pdf.savefig(fig)
        plt.close(fig)
    st.success('Exported the plot to PDF successfully!')
    # Create a download button
    with open(filename, "rb") as f:
        st.download_button(
            label="Download PDF",
            data=f,
            file_name=filename,
            mime="application/pdf"
        )