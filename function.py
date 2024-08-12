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
from matplotlib.patches import Rectangle
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
    title='Heart Rate Over Time'
    try:
        # Ensure the 'value' column is numeric and 'date' column is datetime


        # Check for any rows with NaT in 'date' column and drop them
        
        # Calculate daily mean and quartiles
        daily_stats = df.groupby(df['date'].dt.date)['value'].agg(['mean', 'quantile']).reset_index()
        daily_stats['Q1'] = df.groupby(df['date'].dt.date)['value'].quantile(0.25).values
        daily_stats['Q3'] = df.groupby(df['date'].dt.date)['value'].quantile(0.75).values
        daily_stats.columns = ['date', 'mean', 'quantile', 'Q1', 'Q3']
        daily_stats['date'] = pd.to_datetime(daily_stats['date'])

        # Extend the end date by 2 days
        end_date_extended = pd.to_datetime(end_date)

        # Filter the data between the specified start and end dates
        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date_extended)
        df_filtered = daily_stats[(daily_stats['date'] >= start_date_ts) & (daily_stats['date'] <= end_date_ts)]

        # Apply smoothing with a rolling window
        window_size = 1
        smoothed_mean = df_filtered['mean'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        smoothed_Q1 = df_filtered['Q1'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        smoothed_Q3 = df_filtered['Q3'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')

        # Plot the data
        ax.plot(df_filtered['date'], smoothed_mean, label='Mean', color='magenta')
        ax.fill_between(df_filtered['date'], smoothed_Q1, smoothed_Q3, color='gray', alpha=0.2, label='Q1-Q3 (Interquartile Range)')

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
        ax.legend()
        ax.grid(True)
        ax.set_title(title, loc='left')
        return ax, df_filtered

    except Exception as e:
        print(f"Error: {e}")
        return ax, None

def plot_activity(ax, df, start_date, end_date):
    title='Activity Plot'

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
        ax.set_title(title, loc='left')
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
        ax.xaxis.set_visible(True)
        
        ax.set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
        ax.set_xlim(ax.get_xlim()[0] - pd.Timedelta(days=1), ax.get_xlim()[1] + pd.Timedelta(days=1)) # Add padding
        
        return ax

    except Exception as e:
        print(f"An error occurred: {e}")
        return

# def sleep_graph_ver(ax, slp_df, hrdf, start_date, end_date):
#     title='Sleep graph for in 24 hours'

#     try:
#         # 시간 데이터 변환
#         slp_df['converted_time'] = slp_df['time_stamp'].apply(normalize_time)
#         slp_df['datetime'] = pd.to_datetime(slp_df['date'].dt.date.astype(str) + ' ' + slp_df['converted_time'], errors='coerce')
#         hrdf['converted_time'] = hrdf['time_min'].apply(normalize_time)
#         hrdf['datetime'] = pd.to_datetime(hrdf['date'].dt.date.astype(str) + ' ' + hrdf['converted_time'], errors='coerce')
        
#         # start time 세팅
#         date_range = pd.date_range(start=start_date, end=end_date, freq='T')  # 'T'는 분(minute) 단위를 의미합니다.
        
#         df_based = pd.DataFrame(date_range, columns=['datetime'])

#         # hr setting
#         hrdf_datetimes = set(hrdf['datetime'])
#         df_based['value'] = df_based['datetime'].apply(lambda x: 2 if x in hrdf_datetimes else 1)


#         # 데이터 확장
#         expanded_rows = []
#         for _, row in slp_df.iterrows():
#             start_datetime = row['datetime']
#             duration_seconds = int(row['sleep_duration'])  # 'duration'은 초 단위로 주어짐
#             end_datetime = start_datetime + pd.Timedelta(seconds=duration_seconds)

#             # 시작 시간부터 종료 시간까지 모든 분 생성
#             current_datetime = start_datetime
#             while current_datetime < end_datetime:
#                 expanded_rows.append({
#                     'datetime': current_datetime,
#                     'sleep_stage': row['sleep_stage']
#                 })
#                 current_datetime += pd.Timedelta(minutes=1)

#         # 확장된 데이터 프레임 생성
#         expanded_df = pd.DataFrame(expanded_rows)
#         expand_datetimes = set(expanded_df['datetime'])
#         df_based['events'] = df_based['datetime'].apply(lambda x: 0 if x in expand_datetimes else None)
#         df_based['value'] = df_based['events'].combine_first(df_based['value'])
#         df_based.drop('events', axis=1, inplace=True)

#         # 'datetime' 열을 datetime 객체로 변환
#         df_based['datetime'] = pd.to_datetime(df_based['datetime'])

#         # 데이터를 날짜와 시간으로 분리
#         df_based['date'] = df_based['datetime'].dt.date
#         df_based['hour'] = df_based['datetime'].dt.hour

#         # 유니크한 날짜를 정렬하여 사용
#         unique_dates = df_based['date'].unique()
#         unique_dates.sort()

#         ax.set_xticklabels([])
#         # 각 날짜와 시간대별로 값 플롯
#         for date in unique_dates:
#             day_data = df_based[df_based['date'] == date]
#             for hour in range(24):
#                 hour_data = day_data[day_data['hour'] == hour]
#                 # if not hour_data.empty:
#                 #     value = int(hour_data.iloc[0]['value'])  # 첫 번째 값을 사용하고 정수형으로 변환
#                 #     bars = ax.bar(date, 1, bottom=hour, color=['lightcoral', 'lightgray', 'lightgreen'][value % 3], align='edge')
#                 if not hour_data.empty:
#                     value = int(hour_data.iloc[0]['value'])  # 첫 번째 값을 사용하고 정수형으로 변환
#                     color = ['lightcoral', 'lightgray', 'lightgreen'][value % 3] if value in [0, 1, 2] else 'lightgray'
#                     ax.bar(date, 1, bottom=hour, color=color, align='edge')
        

#         # 그리드 설정
#         ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
#         ax.set_ylim(24, 0)
#         ax.set_xlim(left=pd.to_datetime(start_date), right=pd.to_datetime(end_date))

#         # 축 설정 후 그리드 활성화
#         ax.xaxis.set_major_locator(mdates.DayLocator())
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#         plt.xticks(rotation=45)
#         ax.set_xlabel('Date')
#         ax.set_ylabel('Hour of Day')
#         ax.xaxis.set_visible(False)
#         ax.set_title(title, loc='left')
#         return ax, df_based

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None, None, None

def sleep_graph_ver(ax, slp_df, hrdf, start_date, end_date):
    title='Sleep graph for in 24 hours'

    try:
        # 시간 데이터 변환
        slp_df['converted_time'] = slp_df['time_stamp'].apply(normalize_time)
        slp_df['datetime'] = pd.to_datetime(slp_df['date'].dt.date.astype(str) + ' ' + slp_df['converted_time'], errors='coerce')
        hrdf['converted_time'] = hrdf['time_min'].apply(normalize_time)
        hrdf['datetime'] = pd.to_datetime(hrdf['date'].dt.date.astype(str) + ' ' + hrdf['converted_time'], errors='coerce')

        date_range = pd.date_range(start=start_date, end=end_date, freq='T')  # 'T'는 분(minute) 단위를 의미합니다.
        df_based = pd.DataFrame(date_range, columns=['datetime'])

        # 색상 매핑
        new_cmap = {'rem': 'red', 'light': 'blue', 'deep': 'green', 'awake': 'gray', 'restless': 'orange', 'asleep': 'purple', 'wake': 'yellow', 'missing': 'black'}
        slp_df['sleep_stage'] = slp_df['sleep_stage'].map({'rem': 'rem', 'light': 'light', 'deep': 'deep', 'awake': 'awake', 'restless':'restless'}).fillna('asleep')
        # 초 단위를 버리고 분 단위로 변환
        slp_df['datetime'] = slp_df['datetime'].dt.floor('min')

        # 시간 관련 계산 (분 단위로 변환)
        slp_df['time_minutes'] = slp_df['time_stamp'].apply(lambda x: sum(int(part) * 60 ** (1 - i) for i, part in enumerate(x.split(':'))))
        slp_df['sleep_duration_minutes'] = slp_df['sleep_duration'].astype(float) / 60
        hrdf_datetimes = set(hrdf['datetime'])
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



        # time_minutes가 0인 경우 sleep_duration_minutes도 0으로 설정
        slp_df.loc[slp_df['time_minutes'] == 0, 'sleep_duration_minutes'] = 0

        # 날짜를 넘어가는 경우를 처리하기 위한 데이터 확장
        expanded_rows = []
        for _, row in slp_df.iterrows():
            start_datetime = row['datetime']
            duration_minutes = int(row['sleep_duration_minutes'])
            end_datetime = start_datetime + pd.Timedelta(minutes=duration_minutes)

            # 시작 시간부터 종료 시간까지 모든 분 생성
            current_datetime = start_datetime
            while current_datetime < end_datetime:
                expanded_rows.append({
                    'datetime': current_datetime,
                    'date': current_datetime.date(),
                    'sleep_stage': row['sleep_stage'],
                    'time_minutes': (current_datetime - current_datetime.replace(hour=0, minute=0, second=0)).total_seconds() / 60,
                    'sleep_duration_minutes': 1  # 각 분마다 1분의 sleep_duration_minutes
                })
                current_datetime += pd.Timedelta(minutes=1)

        # 확장된 데이터 프레임 생성
        expanded_df = pd.DataFrame(expanded_rows)
        # 일별 각 단계별 수면시간 계산
        daily_sleep = expanded_df.groupby(['date', 'sleep_stage']).agg({'sleep_duration_minutes': 'sum'}).reset_index()
        daily_sleep_pivot = daily_sleep.pivot(index='date', columns='sleep_stage', values='sleep_duration_minutes').fillna(0)
        # 일별 각 단계별 수면시간을 표로 표시
        st.write("Daily Sleep Duration by Stage")
        st.dataframe(daily_sleep_pivot.T)
        # 심박수 데이터를 수면 데이터와 매핑
        heart_rate_df['heart_rate'] = heart_rate_df['value'].apply(lambda x: 1 if x != -1 else 0)
        heart_rate_df = heart_rate_df[['datetime', 'heart_rate']]
        full_df = pd.merge(expanded_df, heart_rate_df, on='datetime', how='outer').sort_values(by='datetime').fillna({'heart_rate': 0})
        # Compliance 계산
        def calculate_compliance(row):
            if row['heart_rate'] == 1 and row['sleep_stage'] not in ['light', 'rem', 'deep', 'awake', 'restless', 'asleep', 'rem']:
                return 'wake'
            elif row['sleep_stage'] in ['rem']:
                return 'rem'
            elif row['sleep_stage'] in ['light']:
                return 'light'
            elif row['sleep_stage'] in ['deep']:
                return 'deep'
            elif row['sleep_stage'] in ['awake']:
                return 'awake'
            elif row['sleep_stage'] in ['restless']:
                return 'restless'
            elif row['sleep_stage'] in ['asleep']:
                return 'asleep'
            elif row['heart_rate'] == 0:
                return 'missing'
            else:
                return 'wake'  # 만약 다른 상태가 있다면 이를 처리


        full_df['compliance'] = full_df.apply(calculate_compliance, axis=1)
        # 연속적인 compliance 값의 지속 시간을 time_minutes에 추가
        full_df['date_only'] = full_df['datetime'].dt.date
        full_df['sleep_duration_minutes'] = 1
        full_df = full_df.sort_values(by='datetime').reset_index(drop=True)
        # new_df = full_df.groupby(['datetime', 'compliance']).agg({'sleep_duration_minutes': 'sum'}).reset_index()
        # 새로운 데이터프레임 생성
        rows = []
        current_compliance = None
        start_time = None
        total_duration = 0

        for i, row in full_df.iterrows():
            if row['compliance'] != current_compliance:
                if current_compliance is not None:
                    rows.append({
                        'datetime': start_time,
                        'compliance': current_compliance,
                        'sleep_duration_minutes': total_duration,
                        'time_duration': total_duration
                    })
                current_compliance = row['compliance']
                start_time = row['datetime']
                total_duration = row['sleep_duration_minutes']
            else:
                total_duration += row['sleep_duration_minutes']

        # 마지막 행 추가
        rows.append({
            'datetime': start_time,
            'compliance': current_compliance,
            'sleep_duration_minutes': total_duration,
            'time_duration': total_duration
        })
        new_df = pd.DataFrame(rows)


        # Matplotlib을 사용하여 그래프 생성

        for stage, color in new_cmap.items():
            stage_data = new_df[new_df['compliance'] == stage]
            for _, row in stage_data.iterrows():
                rect = Rectangle((mdates.date2num(row['datetime']), row['time_duration']), 1/24, row['sleep_duration_minutes'], color=color)
                ax.add_patch(rect)

        min_date = full_df['datetime'].min()
        max_date = full_df['datetime'].max()

        ax.set_xlim([min_date, max_date])
        ax.set_ylim([0, 24*60])
        ax.set_ylabel('Time (HH:MM)')
        ax.set_xlabel('Date')

        ax.yaxis.set_major_locator(plt.MultipleLocator(60))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60):02}:{int(x%60):02}'))

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.invert_yaxis()
        plt.title('Detailed Sleep Stages Over Time')
        plt.show()

        return ax, df_based

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

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
        st.write(daily_compliance)
        # Calculate daily compliance rate
        daily_compliance_rate = (daily_compliance / 1440) * 100

        # Create a new DataFrame for compliance
        compliance_df = pd.DataFrame({
            'date': daily_compliance_rate.index,
            'compliance_rate': daily_compliance_rate.values
        })
        st.write(compliance_df)
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
        date_index_map = {date: idx for idx, date in enumerate(unique_dates)}
        df['date_index'] = df['date'].map(date_index_map)
        
        st.write(df)
        df['label'] = df['value'].map({0: 'sleep', 1: 'missing', 2: 'wake'})
        
        # 라벨별 갯수 / 60 으로 해당 값을 H 로 치환
        daily_counts = df.groupby(['date', 'label']).size().unstack(fill_value=0)
        daily_counts = (daily_counts / 60).round(1)
        pivot_table = daily_counts.T


        original_col_labels = list(pivot_table.columns)
        col_labels = replace_with_empty_at_interval(original_col_labels)

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