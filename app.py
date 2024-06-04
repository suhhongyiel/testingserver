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
import MatplotlibReportGenerator as mrg
import matplotlib.dates as mdates

# 데이터베이스 연결 설정
db_url = 'mysql+pymysql://root:Korea2022!@119.67.109.156:3306/project_wd'
engine = create_engine(db_url)

# 데이터 검증을 위한 함수
def validate_dataframe(df, expected_columns):
    if df.empty:
        print("DataFrame is empty")
        return False
    for column in expected_columns:
        if column not in df.columns:
            print(f"Missing expected column: {column}")
            return False
    return True

# < === 0값인 날들 회색 처리하는 함수 === >
def shade_zero_data(ax, data_df, data):
    is_zero = data_df == 0
    zero_dates = data[is_zero]['date'].tolist()
    
    if not zero_dates:
        return
    
    start = zero_dates[0]
    for current_date, next_date in zip(zero_dates, zero_dates[1:] + [None]):
        if next_date and (next_date - current_date).days == 1:
            continue
        else:
            ax.axvspan(start, current_date, color='grey', alpha=0.5)
            if next_date:
                start = next_date

# < === 0값인 날들 회색 처리하는 함수 + 결측값 처리 === >
def shade_negative_one_data(ax, data_df, data):
    is_negative_one = data_df == -1
    negative_one_dates = data[is_negative_one]['date'].tolist()
    
    if not negative_one_dates:
        return
    
    start = negative_one_dates[0]
    for current_date, next_date in zip(negative_one_dates, negative_one_dates[1:] + [None]):
        if next_date and (next_date - current_date).days == 1:
            continue
        else:
            ax.axvspan(start, current_date, color='grey', alpha=0.5)
            if next_date:
                start = next_date

# < === 결측치 or 0인 값 => 회색 처리 === >
def shade_negatives_and_zeros(ax, data_series, time_series):
    is_negative_or_zero = (data_series == -1) | (data_series == 0)
    for start, end in zip(time_series[is_negative_or_zero & ~is_negative_or_zero.shift(fill_value=False)], 
                        time_series[is_negative_or_zero & ~is_negative_or_zero.shift(-1, fill_value=False)]):
        ax.axvspan(start, end, color='grey', alpha=0.5)

# < === 문자열 날짜 -> datetime === >
def convert_date(date_str):
    if not isinstance(date_str, str):
        return date_str
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise').strftime('%Y-%m-%d')
    except:
        return pd.to_datetime(date_str, format='%m/%d/%Y', errors='raise').strftime('%Y-%m-%d')

# < === 출력할 데이터 선택 === >
def extract_range_data(uid_table, uid, smcfb_info):
    df = uid_table.copy()
    df['date'] = df['date'].apply(convert_date)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset='date', keep='first')
    
    min_date = df['date'].min()
    max_date = df['date'].max()

    return df, min_date, max_date

# < === PDF 변환 작업 === >
def safe_encode(text, encoding='latin-1', errors='replace'):
    return text.encode(encoding, errors=errors).decode(encoding)

# < 그림 => PDF 변환 및 저장 >
def save_plots_to_pdf_buffer(figs):
    pdf = FPDF()
    
    for fig in figs:
        img_stream = io.BytesIO()
        fig.savefig(img_stream, format="png")
        img_stream.seek(0)

        pdf.add_page()
        pdf.image(img_stream, x=10, y=20, w=190)
    
    buffer = io.BytesIO()
    pdf.output(buffer, "F")
    buffer.seek(0)
    return buffer

# < 주어진 시작/종료 날짜 사이의 모든 시간에 대해 데이터 확인 >
def create_full_datetime_range(df, start_date, end_date):
    full_range = pd.date_range(start=start_date, end=end_date, freq='T')
    full_df = pd.DataFrame(full_range, columns=['datetime'])
    full_df['user_id'] = df['user_id'].iloc[0]

    merged_df = pd.merge(full_df, df, on=['datetime', 'user_id'], how='left')
    return merged_df

def heart_rate_plot(ax, id, start_date, end_date):
    try:
        corrected_id = id.replace('.', '_')
        table_heart_rate = f'{corrected_id}_분별심박수'
        table_resting_heart_rate = f'{corrected_id}_휴식기심박수'

        query_heart_rate = f"""
        SELECT * FROM {table_heart_rate}
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        """
        df = pd.read_sql(query_heart_rate, engine)

        query_resting_heart_rate = f"""
        SELECT * FROM {table_resting_heart_rate}
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        """
        ds = pd.read_sql(query_resting_heart_rate, engine)

        if not validate_dataframe(df, ['date', 'time_min', 'value']) or not validate_dataframe(ds, ['date', 'resting_hr']):
            return None, None, None

        df = df[df['date'] != '-1']
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        ds['date'] = pd.to_datetime(ds['date'], errors='coerce')

        df_filtered = df[(df['time_min'] != '-1') & (df['value'] != -1)]
        df_filtered = df_filtered[(df['date'] >= start_date) & (df['date'] <= end_date)]
        ds_filtered = ds[(ds['date'] >= start_date) & (ds['date'] <= end_date)]

        df_filtered['value'] = pd.to_numeric(df_filtered['value'], errors='coerce')
        ds_filtered['resting_hr'] = pd.to_numeric(ds_filtered['resting_hr'], errors='coerce')

        mean_total = np.mean(df_filtered['value'])
        daily_mean = df_filtered.groupby(df_filtered['date'].dt.date)['value'].mean().mean()

        df_filtered['week'] = df_filtered['date'].dt.to_period('W')
        weekly_daily_mean = df_filtered.groupby('week')['value'].mean().mean()

        max_date = df_filtered['date'].max()
        min_date = max_date - timedelta(days=20)
        df_last_21_days = df_filtered[(df_filtered['date'] >= min_date) & (df_filtered['date'] <= max_date)]

        mean_data = df_last_21_days.groupby(df_last_21_days['date'].dt.date)['value'].mean()
        quartile_data = df_last_21_days.groupby(df_last_21_days['date'].dt.date)['value'].quantile([0.25, 0.75]).unstack()

        window_size = 3
        smoothed_mean = mean_data.rolling(window=window_size, center=True).mean()
        smoothed_q1 = quartile_data[0.25].rolling(window=window_size, center=True).mean()
        smoothed_q3 = quartile_data[0.75].rolling(window=window_size, center=True).mean()

        ax.plot(quartile_data.index, smoothed_mean, label='Mean', color='magenta')
        ax.fill_between(quartile_data.index, smoothed_q1, smoothed_q3, color='gray', alpha=0.2,
                        label='Q1-Q3 (Interquartile Range)')
        
        for index, row in ds_filtered.iterrows():
            if row['resting_hr'] == -1:
                ax.plot(row['date'], 60, 'o', mfc='none', mec='red')
            else:
                ax.plot(row['date'], row['resting_hr'], 'o', color='red')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.set_xlim([min_date, max_date])
        ax.set_xticklabels([])

        ax.set_xlabel('Date')
        ax.set_ylabel('Average Heart Rate')
        ax.legend()
        ax.grid(True)

    except Exception as e:
        print(f"An error occurred: {e}")

    return mean_total, daily_mean, weekly_daily_mean

def plot_activity(ax, id, start_date, end_date):
    try:
        corrected_id = id.replace('.', '_')
        table_activity = f'{corrected_id}_활동량'

        query_activity = f"""
        SELECT * FROM {table_activity}
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        """
        df = pd.read_sql(query_activity, engine)

        if not validate_dataframe(df, ['date', 'steps']):
            return None, None, None

        df = df[df['date'] != '-1']
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        if df_filtered.empty:
            print("No data available in the specified date range.")
            return None, None, None

        df_filtered['steps'] = pd.to_numeric(df_filtered['steps'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['steps'])

        time = df_filtered['date']
        steps = df_filtered['steps']

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

        df_filtered['week'] = df_filtered['date'].dt.isocalendar().week
        weekly_mean_steps = df_filtered.groupby('week')['steps'].mean()
        weekly_mean = weekly_mean_steps.mean()

        return total_steps, daily_mean_steps, weekly_mean

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

def convert_excel_time(excel_time_str, date_str):
    hours, minutes = map(int, excel_time_str.split(':'))
    total_minutes = hours * 60 + minutes
    date_part = parser.parse(date_str)
    time_delta = timedelta(minutes=total_minutes)
    converted_datetime = date_part + time_delta
    return converted_datetime

def expand_sleep_data(row):
    if row['time_stamp'] == '-1' or row['sleep_duration'] == -1:
        return []

    date_str = row['date']
    time_str = row['time_stamp']
    try:
        date_time_str = f"{date_str} {time_str}"
        date_time_obj = parser.parse(date_time_str)
    except ValueError:
        print("ERROR: 날짜 형식을 변환할 수 없습니다.")
        return []

    minutes_list = []
    end_time = date_time_obj + timedelta(seconds=row['sleep_duration'])

    while date_time_obj < end_time:
        minutes_list.append((date_time_obj.strftime('%m/%d/%Y %H:%M'), row['sleep_stage']))
        date_time_obj += timedelta(minutes=1)
    
    return minutes_list

def categorize_row_updated(row):
    if row['SleepStage'] != '0' and row['SleepStage'] != 0:
        return 'Asleep'
    elif pd.notna(row['valid']) and row['valid'] == 1:
        return 'Wake'
    elif pd.notna(row['valid']) and row['valid'] == 0:
        return 'Non-compliant'
    else:
        return 'Wake'

def sleep_graph_ver(ax, id, start_date, end_date, nan_value_df):
    try:
        end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        corrected_id = id.replace('.', '_')
        table_sleep_detail = f'{corrected_id}_수면상세'

        query_sleep_detail = f"""
        SELECT * FROM {table_sleep_detail}
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        """
        sleep_detail_df = pd.read_sql(query_sleep_detail, engine)
        
        if not validate_dataframe(sleep_detail_df, ['date', 'time_stamp', 'sleep_duration', 'sleep_stage']):
            return None, None, None

        sleep_detail_df['date'] = sleep_detail_df['date'].str.lstrip('0').str.replace('/0', '/')
        
        sleep_minutes = sleep_detail_df.apply(expand_sleep_data, axis=1).sum()

        sleep_minutes_df = pd.DataFrame(sleep_minutes, columns=['DateTime', 'SleepStage'])
        sleep_minutes_df['DateTime'] = pd.to_datetime(sleep_minutes_df['DateTime'], format='%m/%d/%Y %H:%M')

        min_date = sleep_minutes_df['DateTime'].min().replace(hour=0, minute=0)
        actual_max_date = sleep_minutes_df['DateTime'].max()

        corrected_min_date = min_date.replace(day=1)
        corrected_all_minutes = pd.date_range(start=corrected_min_date, end=actual_max_date, freq='T').to_series()
        corrected_all_minutes_df = pd.DataFrame(corrected_all_minutes, columns=['DateTime'])

        corrected_full_sleep_data = pd.merge(corrected_all_minutes_df, sleep_minutes_df, on='DateTime', how='left')
        # nan_value_df.to_csv('./nana.csv')
        corrected_full_sleep_data['SleepStage'].fillna(0, inplace=True)
        corrected_sleep_data_cleaned = corrected_full_sleep_data.copy()
        corrected_sleep_data_cleaned.loc[:, 'time_min'] = corrected_sleep_data_cleaned['DateTime'].dt.strftime('%H:%M:%S')
        corrected_sleep_data_cleaned.loc[:, 'date'] = corrected_sleep_data_cleaned['DateTime'].dt.strftime('%m/%d/%Y')
        # corrected_sleep_data_cleaned.to_csv('./sleep_.csv')

        merged_data = pd.merge(corrected_sleep_data_cleaned, nan_value_df[['date', 'time_min', 'valid']], 
                               on=['date', 'time_min'], how='left')

        merged_data = merged_data[(merged_data['DateTime'] >= start_date) & (merged_data['DateTime'] <= end_date)]
        # merged_data.to_csv("./merged.csv")

        merged_data['Category'] = merged_data.apply(categorize_row_updated, axis=1)
        merged_data['date'] = merged_data['DateTime'].dt.date
        merged_data['hour'] = merged_data['DateTime'].dt.hour
        merged_data['minute'] = merged_data['DateTime'].dt.minute

        categories_priority = ['Asleep', 'Non-compliant', 'Wake']
        pivot_data_detailed = merged_data.pivot_table(index=['hour', 'minute'], columns='date', values='Category', aggfunc='first')
        pivot_data_priority_replaced = pivot_data_detailed.replace(categories_priority, range(len(categories_priority)))

        colors = ['blue', 'red', 'green']
        cmap = plt.cm.colors.ListedColormap(colors)
        
        cbar = ax.imshow(pivot_data_priority_replaced, aspect='auto', cmap=cmap)
        y_ticks = [0, 6*60, 12*60, 18*60]
        y_tick_labels = ['00:00', '06:00', '12:00', '18:00']
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        
        ax.set_xticks(np.arange(len(pivot_data_detailed.columns)))
        ax.set_xticklabels([])

        for i in range(len(pivot_data_detailed.columns) - 1):
            ax.axvline(x=i+0.5, color='white', linestyle='--', linewidth=0.5)

        ax.set_ylabel('Time of Day')
        ax.set_xlabel('Date')
        ax.set_xticklabels([])

        return start_date, end_date, merged_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

def plot_compliance(ax, id, start_date, end_date, dffff):
    try:
        corrected_id = id.replace('.', '_')
        table_heart_rate = f'{corrected_id}_분별심박수'

        query_heart_rate = f"""
        SELECT * FROM `{table_heart_rate}`
        WHERE `date` BETWEEN '{start_date}' AND '{end_date}'
        """
        df = pd.read_sql(query_heart_rate, engine)

        if not validate_dataframe(df, ['date', 'time_min', 'value']):
            return None, None, None, pd.DataFrame()

        df = df[df['time_min'] != '-1']
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time_min'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df_filtered = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] < pd.to_datetime(end_date))]

        if df_filtered.empty:
            print("Filtered data is empty.")
            return (None, None, None, pd.DataFrame())

        full_df_list = [create_full_datetime_range(group, start_date, end_date) for _, group in df_filtered.groupby(['user_id', df_filtered['datetime'].dt.date])]
        if not full_df_list:
            print("No full datetime ranges available after processing.")
            return (None, None, None, pd.DataFrame())

        df_complete = pd.concat(full_df_list).reset_index(drop=True)
        df_complete['valid'] = df_complete['value'].notna().astype(int)
        daily_compliance = df_complete.groupby(df_complete['datetime'].dt.date)['valid'].sum()

        if daily_compliance.empty:
            print("No compliance data to plot.")
            return (start_date, 0, 0, df_complete)

        daily_compliance.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_xlabel('Date')
        ax.set_ylabel('Compliance Count')
        ax.set_xticklabels([])
        
        date_mean = daily_compliance.mean()
        date_total = daily_compliance.sum()
        return (start_date, date_mean, date_total, df_complete)
    except Exception as e:
        print(f"Error during plot_compliance: {e}")
        return (None, None, None, pd.DataFrame())

# < === 페이지 레이아웃 === >
def page_about():
    st.title("DataBase")
    try:
        query = "SELECT study_ID FROM fitbit_device_list"
        device_info_options = [row[0] for row in pd.read_sql(query, engine).values]
    except SQLAlchemyError as e:
        st.error(f"An error occurred: {e}")
        return
    
    device_info = st.sidebar.selectbox("Fitbit ID:", device_info_options, None)
    if device_info is not None:
        device_info = device_info.replace('_', '.')

    start_date = st.sidebar.date_input("Start Date", None)
    last_date = st.sidebar.date_input("Last Date", None)

    st.write("start_date: ", start_date)
    st.write("last_date: ", last_date)

    if start_date and last_date:
        start_date = datetime.combine(start_date, datetime.min.time())
        last_date = datetime.combine(last_date, datetime.min.time())
        date_difference = (last_date - start_date).days
    else:
        date_difference = 0
        st.warning("Please select both start date and last date.")

    initial = st.sidebar.text_input("Patient info(Initial)", None)
    age = st.sidebar.text_input("Age", None)
    sex = st.sidebar.radio("Sex", ["Male", "Female"], None)
    cancer_type_options = ["Breast", "Lung", "Prostate", "Colon", "Skin", "Other"]
    cancer_type = st.sidebar.selectbox("Cancer Type", cancer_type_options, None)

    treatment_type_option = ["1","2","3","4"]
    treatment_type = st.sidebar.selectbox("Treatment Type", treatment_type_option)

    plt.style.use('default')
    plt.rcParams['axes.grid'] = False

    fig, axs = plt.subplots(6, 1, figsize=(10, 15), constrained_layout=True)

    if device_info and start_date and last_date and age and sex and cancer_type and treatment_type:
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            import matplotlib.font_manager as fm
            font_path = 'NanumGothic-Regular.ttf'

            font_prop = fm.FontProperties(fname=font_path)

            demographic = np.array([
                [f"Fitbit ID: {device_info}"],
                [f"Data extraction date: {datetime.today().strftime('%Y-%m-%d')}"],
                [f"Tracking date: {start_date.strftime('%Y-%m-%d')} (Start date) - {last_date.strftime('%Y-%m-%d')} (Last date)"],
                [f"Patient info: {initial}, Age: {age}Y, Sex: {sex}"],
                [f"Cancer type: {cancer_type}"],
                [f"Treatment Type: {treatment_type}"]
            ])

            table = axs[0].table(cellText=demographic, loc='center', cellLoc='left', bbox=[-0.1, 0, 0.6, 1], edges='open')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            axs[0].axis("off")

            a, date_mean, date_total, missing_real_merged = plot_compliance(axs[1], device_info, start_date, last_date, date_difference)

            if date_mean is None or date_total is None:
                st.error("Failed to calculate compliance data.")
            else:
                axs[1].set_title(
                    f'Compilance \n'
                    f'(1) total wearing time {round(date_total, 2)}      '
                    f'(2) daily average wearing time {round(date_mean, 2)}',
                    fontdict={'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 10}, loc="left")

            total_sample, day_mean_sample, week_mean = plot_activity(axs[2], device_info, start_date, last_date)
            if day_mean_sample is not None and week_mean is not None:
                fontinfo = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 10}
                axs[2].set_title(
                    'Physical activity (steps)\n'
                    f'(1) weekly average steps: {round(week_mean, 2)}        '
                    f'(2) daily average steps: {round(day_mean_sample, 2)}',
                    fontdict=fontinfo, loc="left")
            else:
                st.error("Failed to calculate physical activity data.")

            m, d, w = heart_rate_plot(axs[3], device_info, start_date, last_date)
            if m is not None and d is not None:
                fontinfo = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 10}
                axs[3].set_title('Daily Heart Rate\n'
                                 f'(1) Total Average HR: {round(m, 2)} bpm      '
                                 f'(2) Daily Average HR: {round(d, 2)} bpm',
                                 fontdict=fontinfo, loc="left")
            else:
                print("Failed to calculate heart rate data or data is missing.")

            s, e, d = sleep_graph_ver(axs[4], device_info, start_date, last_date, missing_real_merged)
            if d is None:
                st.error("Failed to calculate sleep data.")
                return
            
            ds = d.groupby("date")

            date_range = pd.date_range(start_date, last_date)
            day_values = [[ds.month, ds.day] for ds in date_range]
            day_list = []
            for i in range(len(day_values)):
                day_list.append(f"{day_values[i][0]}/{day_values[i][1]}")

            cols = [i for i in day_list]
            tmp1 = []
            tmp2 = []
            tmp3 = []
            summary = {
                'Sleep Time': [],
                'Not Worn Time': [],
                'Wake': []
            }
            for name, group in d.groupby('date'):
                summary['Sleep Time'].append(group[group['SleepStage'] != 0].shape[0])
                summary['Not Worn Time'].append(group[(group['valid'] == 0) & (group['SleepStage'] == 0)].shape[0])
                summary['Wake'].append(group[group['SleepStage'] == 0].shape[0])

            for i in range(len(summary['Sleep Time'])):
                tmp1.append(summary['Sleep Time'][i])
                tmp2.append(summary['Not Worn Time'][i])
                tmp3.append(summary['Wake'][i])

            all_tmp = [tmp1, tmp2, tmp3]
            result = np.array(all_tmp)

            colcolors = np.array([[0.2, 0.6, 1.0]] * 22)
            rowcolors = np.array([
                [1.0, 0.3, 0.3, 0.7],
                [1.0, 1.0, 0.2, 0.7],
                [0, 1.0, 0, 0.7]
            ])

            rows = ["Sleep", "N/A", "Wake"]
            axs[4].set_title('Sleep Detail', fontdict=fontinfo, loc="left")

            axs[5].table(cellText=result,
                         loc='center',
                         cellLoc='center',
                         bbox=[0, 0, 1, 1],
                         rowLabels=rows,
                         rowColours=rowcolors,
                         colLabels=cols,
                         colColours=colcolors
                         )

            text_contents = [
                f"Fitbit ID: {device_info.replace('_', '.')}",
                f"Data extraction date: {datetime.today()}",
                f"Tracking date: {start_date.strftime('%Y-%m-%d')} - {last_date.strftime('%Y-%m-%d')}",
                f"Patient info {initial}, Age: {age}Y, Sex: {sex}",
                f"Cancer type: {cancer_type}",
                f"Treatment Type: {treatment_type}"
            ]
            st.pyplot(fig)

            pdf.savefig(fig)
            plt.close(fig)

        pdf_buffer.seek(0)

        st.download_button(
            label='Download PDF',
            data=pdf_buffer,
            file_name=f"{device_info}.pdf",
            mime='application/pdf'
        )

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

# Test the function
# extract_range_data(["file1.csv", "file2.csv", ...], "sample_uid", [start_date, end_date])

# 필요없음
# def plot_resting(df, min_date, max_date):
#     fig, ax = plt.subplots(figsize=(12, 6))
#     axes = [ax] 

#     data_sorted = df.sort_values(by='date')

#     time = data_sorted['date']
#     sample = data_sorted['resting_hr'] 


#     axes[0].plot(time, sample, '-o', label="resting_hr", color='black') # 정상 상황
#     shade_negative_one_data(axes[0], data_sorted['resting_hr'], data_sorted) # 결측값 처리
#     axes[0].set_title('resting_hr')
#     axes[0].set_ylabel('Value')
    

#     # x축 날짜 형식 설정
#     for ax in axes:
#         ax.set_xlim(min_date, max_date)
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#         ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1)) # 1주일 단위
#         ax.tick_params(axis='x', rotation=45)
        
#     plt.tight_layout()
#     return fig

# 필요없음
# def plot_activity(df, min_date, max_date):

#     # PLOT GRAPH
#     fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
#     data_sorted = df.sort_values(by='date')
#     sample = data_sorted['distance'] 
#     time = data_sorted['date']
#     # Set the title for the entire plot
#     plt.suptitle(f"{data_sorted['user_id'][0]}", fontsize=16)  # Assuming file_name is something like "name.csv"
#     # distance 에대한 plot
#     axes[0].plot(time, sample, '-o', label="Distance", color='blue')
#     shade_zero_data(axes[0], data_sorted['distance'], data_sorted)

#     axes[0].set_title('Distance')
#     axes[0].set_ylabel('Value')

#     # Steps 에대한 plot
#     sample = data_sorted['steps']
    
#     axes[1].plot(time, sample, '-o', label="Steps", color='navy')
#     shade_zero_data(axes[1], data_sorted['steps'], data_sorted)
#     axes[1].set_title('Steps')
#     axes[1].set_ylabel('Value')

#     # calories 에 대한 plot
#     sample = data_sorted['calories']
    
#     axes[2].plot(time, sample, '-o', label="calories", color='darkblue')
#     shade_zero_data(axes[2], data_sorted['calories'], data_sorted)
#     axes[2].set_title('calories')
#     axes[2].set_ylabel('Value')

#     plt.tight_layout()
#     return fig

# 필요없음
# def plot_sleeping(df, min_date, max_date):
# # extract directory    

#     time_series = df['date']

#     fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
#     data_sorted = df.sort_values(by='date')
#     plt.suptitle(f"{data_sorted['user_id'][0]}", fontsize=16)  # Assuming file_name is something like "name.csv"
#     # Main sleep data plot
#     axes[0].plot(data_sorted['date'], df['totalMinutesAsleep'], '-o', label='Total Minutes Asleep', color='blue')
#     axes[0].plot(data_sorted['date'], df['totalTimeInBed'], '-o', label='Total Time in Bed', color='orange')
#     shade_negatives_and_zeros(axes[0], df['totalMinutesAsleep'], time_series)
#     axes[0].legend()
#     axes[0].set_title("Main Sleep Data")

#     # Sleep stages stacked bar chart
#     axes[1].bar(data_sorted['date'], df['stages_deep'], label='Deep Sleep', color='darkblue')
#     axes[1].bar(data_sorted['date'], df['stages_light'], label='Light Sleep', color='lightblue', bottom=df['stages_deep'])
#     axes[1].bar(data_sorted['date'], df['stages_rem'], label='REM Sleep', color='green', 
#         bottom=df['stages_deep'] + df['stages_light'])
#     axes[1].bar(data_sorted['date'], df['stages_wake'], label='Awake Time', color='red', 
#         bottom=df['stages_deep'] + df['stages_light'] + df['stages_rem'])
#     axes[1].legend()
#     axes[1].set_title("Sleep Stages")

#     # Efficiency plot
#     axes[2].plot(data_sorted['date'], df['efficiency'], '-o', color='darkcyan', label='Sleep Efficiency')
#     shade_negatives_and_zeros(axes[2], df['efficiency'], data_sorted['date'])
#     axes[2].legend()
#     axes[2].set_title("Sleep Efficiency")

#     # Total sleep records plot
#     axes[3].bar(data_sorted['date'], df['totalSleepRecords'], color='darkgreen', label='Sleep Records')
#     shade_negatives_and_zeros(axes[3], df['totalSleepRecords'], data_sorted['date'])
#     axes[3].legend()
#     axes[3].set_title("Total Sleep Records")

#     # Adjusting the x-ticks for better visibility
#     for ax in axes:
#         ax.tick_params(axis='x', rotation=45)

#     plt.tight_layout()
#     return fig


# 다운로드 페이지
# def page_download():
#     st.write("download")
#     # 선택 하면 전체 plot 출력
#     st.title("DataBase")
#     # Add content for the about page
#     with db.cursor() as cursor:
#         cursor.execute("SELECT study_ID FROM fitbit_device_list")
#         device_info_options = [row[0] for row in cursor.fetchall()]
#     device_info = st.selectbox("SELECT Device Info", device_info_options)

#     table_name = f"{device_info}"
#     st.info(f"Selected Device Info: {device_info}")
#     st.info(f"Table Name: {table_name}")

#     st.write("this is table name: ", table_name)

#     all_name = get_table_names(table_name)

#     # st.write("this is all name: ", all_name)

#     st.write(all_name)
#     # AZM_data = get_table_data(all_name[0])
#     # df, min_date, max_date = extract_range_data(AZM_data, device_info, all_name[0])

#     # AZM_data = get_table_data(all_name[2])
#     # df, min_date, max_date = extract_range_data(AZM_data, device_info, all_name[2])

#     Sleep_data = get_table_data(all_name[4])
#     df4, min_date4, max_date4 = extract_range_data(Sleep_data, device_info, all_name[4])
#     psleep = plot_sleeping(df4, min_date4, max_date4)
#     st.pyplot(psleep)

#     Activity_data = get_table_data(all_name[5])
#     df5, min_date5, max_date5 = extract_range_data(Activity_data, device_info, all_name[5])
#     pact = plot_activity(df5, min_date5, max_date5)
#     st.pyplot(pact)

#     resting_heart_data = get_table_data(all_name[6])
#     df6, min_date6, max_date6 = extract_range_data(resting_heart_data, device_info, all_name[6])
#     p1 = plot_resting(df6, min_date6, max_date6)
#     st.pyplot(p1)


#     figs = [psleep, pact, p1]
    
#     buffer = save_plots_to_pdf_buffer(figs)
#     st.download_button(
#         label="Download PDF",
#         data=buffer,
#         file_name=f"{device_info}.pdf",
#         mime="application/pdf",
#     )
