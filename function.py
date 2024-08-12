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
from matplotlib.table import Table

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
    title = 'Heart Rate Over Time'
    try:
        # Ensure the 'value' column is numeric and 'date' column is datetime
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Check for any rows with NaT in 'date' column and drop them
        df = df.dropna(subset=['date'])

        # Calculate daily mean and quartiles
        daily_stats = df.groupby(df['date'].dt.date)['value'].agg(['mean']).reset_index()
        daily_stats['Q1'] = df.groupby(df['date'].dt.date)['value'].quantile(0.25).values
        daily_stats['Q3'] = df.groupby(df['date'].dt.date)['value'].quantile(0.75).values
        daily_stats.columns = ['date', 'mean', 'Q1', 'Q3']
        daily_stats['date'] = pd.to_datetime(daily_stats['date'])

        # 날짜의 중앙값으로 이동
        daily_stats['date_mid'] = daily_stats['date'] + pd.Timedelta(hours=12)

        # Extend the end date by 2 days
        end_date_extended = pd.to_datetime(end_date)

        # Filter the data between the specified start and end dates
        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date_extended)
        df_filtered = daily_stats[(daily_stats['date_mid'] >= start_date_ts) & (daily_stats['date_mid'] <= end_date_ts)]

        # Apply smoothing with a rolling window
        window_size = 1
        smoothed_mean = df_filtered['mean'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        smoothed_Q1 = df_filtered['Q1'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        smoothed_Q3 = df_filtered['Q3'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')

        # Plot the data using 중앙값을 이동한 날짜 사용
        ax.plot(df_filtered['date_mid'], smoothed_mean, label='Mean', color='magenta')
        ax.fill_between(df_filtered['date_mid'], smoothed_Q1, smoothed_Q3, color='gray', alpha=0.2, label='Q1-Q3 (Interquartile Range)')

        # Set y-axis limit
        ax.set_ylim(0, 200)

        # Format the x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.set_xlim([start_date_ts, end_date_ts])
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Set labels and legend
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Heart Rate')
        ax.xaxis.set_visible(False)
        ax.legend()
        ax.grid(True)
        ax.set_title(title, loc='left')
        return ax, df_filtered

    except Exception as e:
        print(f"Error: {e}")
        return ax, None

def plot_activity(ax, df, start_date, end_date):
    title = 'Activity Plot'

    try:
        df['date'] = pd.to_datetime(df['date'])
        df = df.groupby('date', as_index=False)['steps'].sum()
        df.sort_values('date', inplace=True)
        
        df['steps'] = pd.to_numeric(df['steps'], errors='coerce')
        df = df.dropna(subset=['steps'])
        
        # 날짜의 중앙값 계산
        df['date_mid'] = df['date'] + pd.Timedelta(hours=12)
        
        time = df['date_mid']  # 중앙값으로 설정된 시간을 사용
        steps = df['steps']

        total_steps = steps.sum()
        daily_mean_steps = steps.mean()
        ax.set_title(title, loc='left')
        ax.plot(time, steps, '-o', label="Steps", color='navy')
        ax.set_ylabel('Steps')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels([])

        for date in df['date']:
            ax.axvline(x=date, color='grey', linestyle='--', alpha=0.5)

        df['week'] = df['date'].dt.isocalendar().week
        weekly_mean_steps = df.groupby('week')['steps'].mean()
        weekly_mean = weekly_mean_steps.mean()
        ax.xaxis.set_visible(True)
        
        ax.set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
        ax.set_xlim(ax.get_xlim()[0] - pd.Timedelta(days=1), ax.get_xlim()[1] + pd.Timedelta(days=1))  # Add padding
        
        return ax

    except Exception as e:
        print(f"An error occurred: {e}")
        return

def sleep_graph_ver(ax, slp_df, hrdf, start_date, end_date):
    title='Sleep graph for in 24 hours'

    try:
        # 시간 데이터 변환
        slp_df['converted_time'] = slp_df['time_stamp'].apply(normalize_time)
        slp_df['datetime'] = pd.to_datetime(slp_df['date'].dt.date.astype(str) + ' ' + slp_df['converted_time'], errors='coerce')

        hrdf['converted_time'] = hrdf['time_min'].apply(normalize_time)
        hrdf['datetime'] = pd.to_datetime(hrdf['date'].dt.date.astype(str) + ' ' + hrdf['converted_time'], errors='coerce')
        
        # start time 세팅
        date_range = pd.date_range(start=start_date, end=end_date, freq='T')  # 'T'는 분(minute) 단위를 의미합니다.
        
        df_based = pd.DataFrame(date_range, columns=['datetime'])

        # hr setting: missing 찾는 코드 2면 ok 1 이 면 missing
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
                current_datetime = current_datetime.replace(second=0)
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
        df_based['min'] = df_based['datetime'].dt.minute

        # 유니크한 날짜를 정렬하여 사용
        unique_dates = df_based['date'].unique()
        unique_dates.sort()

        ax.set_xticklabels([])
        # 각 날짜와 시간대별로 값 플로팅
        for date in unique_dates:
            day_data = df_based[df_based['date'] == date]
            for hour in range(24):
                hour_data = day_data[day_data['hour'] == hour]

                if not hour_data.empty:
                    # 각 hour 내의 고유한 value 값을 가져옴
                    for _, row in hour_data.iterrows():
                        minute = row['min']
                        value = int(row['value'])
                        color = ['lightcoral', 'lightgray', 'lightgreen'][value % 3] if value in [0, 1, 2] else 'lightgray'

                        # 1시간을 60분으로 나누어 해당 분 위치에 바를 그림
                        ax.bar(date, 1/60, bottom=hour + minute/60, color=color, align='edge')

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
        ax.set_title(title, loc='left')
        return ax, df_based

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

def plot_compliance(ax, df, start_date, end_date):
    try:
        # Combine date and time to create a full datetime column
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time_min'])
        df = df.dropna(subset=['datetime'])

        st.write(end_date)

        # Drop duplicates based on user_id and datetime
        df = df.drop_duplicates(subset=['user_id', 'datetime'], keep='first')

        # Create a 'valid' column for compliance check
        df['valid'] = (df['value'] != -1) & (df['value'] != 0)
        df['valid'] = df['valid'].astype(int)

        # Group by date and calculate compliance
        df['date'] = df['datetime'].dt.date
        daily_compliance = df.groupby('date')['valid'].sum()
        daily_total = df.groupby('date')['valid'].count()
        # Calculate daily compliance rate
        daily_compliance_rate = (daily_compliance / 1440) * 100

        # Create a new DataFrame for compliance
        compliance_df = pd.DataFrame({
            'date': daily_compliance_rate.index,
            'compliance_rate': daily_compliance_rate.values
        })
        # Plot the daily compliance rate
        compliance_df.set_index('date')['compliance_rate'].plot(kind='bar', color='skyblue', ax=ax)

        ax.set_xlabel('Date')
        ax.set_ylabel('Compliance Rate (%)')
        ax.set_title('Daily Compliance', loc='left')

        # Hide x-axis labels to avoid clutter
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
        ax.text(0.0, 1 - 0.2 * i, text, transform=ax.transAxes, va='top', ha='left', fontdict=fontdict)

    ax.axis('off')  # Hide the axes
    return ax

#5번째 값마다 빈 문자열로 대체하는 함수
def replace_with_empty_at_interval(labels, interval=5):
    for i in range(len(labels)):
        if (i + 1) % interval != 0:
            labels[i] = ''
    return labels 

def sleep_table_area(ax, df, start_date, end_date):
    try:
        # 데이터 로드 및 날짜 필터링
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 날짜별 라벨 집계
        df['date'] = df['datetime'].dt.date
        unique_dates = sorted(df['date'].unique())
        
        # 날짜를 'day/month' 형식으로 변환
        formatted_dates = [date.strftime('%m/%d') for date in unique_dates]
        date_index_map = {date: idx for idx, date in enumerate(unique_dates)}
        df['date_index'] = df['date'].map(date_index_map)

        df['label'] = df['value'].map({0: 'sleep', 1: 'missing', 2: 'wake'})
        
        # 라벨별 갯수 / 60 으로 해당 값을 H 로 치환
        daily_counts = df.groupby(['date', 'label']).size().unstack(fill_value=0)
        daily_counts = (daily_counts / 60).round(1)
        pivot_table = daily_counts.T
        
        original_col_labels = list(pivot_table.columns)
        col_labels = replace_with_empty_at_interval(formatted_dates)

        # 피벗 테이블을 텍스트 테이블로 플롯
        table = ax.table(cellText=pivot_table.values, colLabels=col_labels, rowLabels=pivot_table.index, loc='center')
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
        for (i, key) in enumerate(pivot_table.index):
            for j in range(len(col_labels)):
                table[(i+1, j)].set_facecolor(colors.get(key, 'white'))

        ax.axis('off')  # 축 비활성화

    except Exception as e:
        print(f"An error occurred: {e}")
    
    return ax
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