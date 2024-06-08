import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime as dt
import pymysql
import pandas as pd
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from dateutil import parser
from datetime import datetime
import io
from fpdf import FPDF



def shade_zero_data(ax, data_df, data): 
    is_zero = data_df == 0
    zero_dates = data[is_zero]['date'].tolist()
    
    if not zero_dates:
        return
    
    start = zero_dates[0]
    for current_date, next_date in zip(zero_dates, zero_dates[1:] + [None]):
        if next_date and (next_date - current_date).days == 1:  # If they are consecutive
            continue
        else:
            ax.axvspan(start, current_date, color='grey', alpha=0.5)
            if next_date:
                start = next_date

def shade_negative_one_data(ax, data_df, data):
    is_negative_one = data_df == -1
    negative_one_dates = data[is_negative_one]['date'].tolist()
    
    if not negative_one_dates:
        return
    
    start = negative_one_dates[0]
    for current_date, next_date in zip(negative_one_dates, negative_one_dates[1:] + [None]):
        if next_date and (next_date - current_date).days == 1:  # If they are consecutive
            continue
        else:
            ax.axvspan(start, current_date, color='grey', alpha=0.5)
            if next_date:
                start = next_date

def convert_date(date_str):
    if not isinstance(date_str, str):
        return date_str
    try:
        # First, try parsing with ISO format
        return pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise').strftime('%Y-%m-%d')
    except:
        # If the above fails, try parsing with %m/%d/%Y format
        return pd.to_datetime(date_str, format='%m/%d/%Y', errors='raise').strftime('%Y-%m-%d')

def extract_range_data(uid_table, uid, smcfb_info):

    df = uid_table
    # df = uid_table deep copy를 생성하여 원본 데이터에 영향을 주지 않습니다.
    df = uid_table.copy()

    # 날짜 변환 전에 date 열의 고유한 값들을 확인합니다.
    unique_dates = df['date'].unique()
    # Convert the 'date' column values to datetime objects    
    # Now format the datetime values to the desired string format
    df['date'] = df['date'].apply(convert_date)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset='date', keep='first')
    
    
    min_date = df['date'].min()
    max_date = df['date'].max()

    print(min_date)
    print(max_date)

    return df, min_date, max_date

def shade_negatives_and_zeros(ax, data_series, time_series):
    is_negative_or_zero = (data_series == -1) | (data_series == 0)
    for start, end in zip(time_series[is_negative_or_zero & ~is_negative_or_zero.shift(fill_value=False)], 
                        time_series[is_negative_or_zero & ~is_negative_or_zero.shift(-1, fill_value=False)]):
        ax.axvspan(start, end, color='grey', alpha=0.5)

# Test the function
# extract_range_data(["file1.csv", "file2.csv", ...], "sample_uid", [start_date, end_date])

def plot_resting(df, min_date, max_date):
    fig, ax = plt.subplots(figsize=(12, 6))
    axes = [ax] 

    data_sorted = df.sort_values(by='date')

    time = data_sorted['date']
    sample = data_sorted['resting_hr'] 


    axes[0].plot(time, sample, '-o', label="resting_hr", color='black')
    shade_negative_one_data(axes[0], data_sorted['resting_hr'], data_sorted)
    axes[0].set_title('resting_hr')
    axes[0].set_ylabel('Value')
    

    # x축 날짜 형식 설정
    for ax in axes:
        ax.set_xlim(min_date, max_date)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    return fig

def plot_activity(df, min_date, max_date):

    # PLOT GRAPH
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    data_sorted = df.sort_values(by='date')
    sample = data_sorted['distance'] 
    time = data_sorted['date']
    # Set the title for the entire plot
    plt.suptitle(f"{data_sorted['user_id'][0]}", fontsize=16)  # Assuming file_name is something like "name.csv"
    # distance 에대한 plot
    axes[0].plot(time, sample, '-o', label="Distance", color='blue')
    shade_zero_data(axes[0], data_sorted['distance'], data_sorted)

    axes[0].set_title('Distance')
    axes[0].set_ylabel('Value')

    # Steps 에대한 plot
    sample = data_sorted['steps']
    
    axes[1].plot(time, sample, '-o', label="Steps", color='navy')
    shade_zero_data(axes[1], data_sorted['steps'], data_sorted)
    axes[1].set_title('Steps')
    axes[1].set_ylabel('Value')

    # calories 에 대한 plot
    sample = data_sorted['calories']
    
    axes[2].plot(time, sample, '-o', label="calories", color='darkblue')
    shade_zero_data(axes[2], data_sorted['calories'], data_sorted)
    axes[2].set_title('calories')
    axes[2].set_ylabel('Value')

    plt.tight_layout()
    return fig

def plot_sleeping(df, min_date, max_date):
# extract directory    

    time_series = df['date']

    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    data_sorted = df.sort_values(by='date')
    plt.suptitle(f"{data_sorted['user_id'][0]}", fontsize=16)  # Assuming file_name is something like "name.csv"
    # Main sleep data plot
    axes[0].plot(data_sorted['date'], df['totalMinutesAsleep'], '-o', label='Total Minutes Asleep', color='blue')
    axes[0].plot(data_sorted['date'], df['totalTimeInBed'], '-o', label='Total Time in Bed', color='orange')
    shade_negatives_and_zeros(axes[0], df['totalMinutesAsleep'], time_series)
    axes[0].legend()
    axes[0].set_title("Main Sleep Data")

    # Sleep stages stacked bar chart
    axes[1].bar(data_sorted['date'], df['stages_deep'], label='Deep Sleep', color='darkblue')
    axes[1].bar(data_sorted['date'], df['stages_light'], label='Light Sleep', color='lightblue', bottom=df['stages_deep'])
    axes[1].bar(data_sorted['date'], df['stages_rem'], label='REM Sleep', color='green', 
        bottom=df['stages_deep'] + df['stages_light'])
    axes[1].bar(data_sorted['date'], df['stages_wake'], label='Awake Time', color='red', 
        bottom=df['stages_deep'] + df['stages_light'] + df['stages_rem'])
    axes[1].legend()
    axes[1].set_title("Sleep Stages")

    # Efficiency plot
    axes[2].plot(data_sorted['date'], df['efficiency'], '-o', color='darkcyan', label='Sleep Efficiency')
    shade_negatives_and_zeros(axes[2], df['efficiency'], data_sorted['date'])
    axes[2].legend()
    axes[2].set_title("Sleep Efficiency")

    # Total sleep records plot
    axes[3].bar(data_sorted['date'], df['totalSleepRecords'], color='darkgreen', label='Sleep Records')
    shade_negatives_and_zeros(axes[3], df['totalSleepRecords'], data_sorted['date'])
    axes[3].legend()
    axes[3].set_title("Total Sleep Records")

    # Adjusting the x-ticks for better visibility
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


# 페이지
def page_about():
    st.title("DataBase")
    # Add content for the about page
    with db.cursor() as cursor:
        cursor.execute("SELECT study_ID FROM fitbit_device_list")
        device_info_options = [row[0] for row in cursor.fetchall()]

    device_info = st.selectbox("SELECT Device Info", device_info_options)

    table_name = f"{device_info}"
    st.info(f"Selected Device Info: {device_info}")
    st.info(f"Table Name: {table_name}")
    all_name = get_table_names(table_name)
    
    smcfb_info = st.selectbox("SELECT SMCFB Info", all_name)
    # st.write(smcfb_info)
    smc_info = get_table_data(smcfb_info)
    # st.write(smc_info)

    if smcfb_info in table_name + "_휴식기심박수":
        df, min_date, max_date = extract_range_data(smc_info, device_info, smcfb_info)
        p1 = plot_resting(df, min_date, max_date)
        st.pyplot(p1)

    elif smcfb_info in table_name + "_활동량":
        df, min_date, max_date = extract_range_data(smc_info, device_info, smcfb_info)
        p2 = plot_activity(df, min_date, max_date)
        st.pyplot(p2)

    elif smcfb_info in table_name + "_AZM분별활동":
        st.write("gg1")

    elif smcfb_info in table_name + "_분별심박수":
        st.write("gg2")

    elif smcfb_info in table_name + "_수면상세":
        st.write("gg3")

    elif smcfb_info in table_name + "_수면요약":
        df, min_date, max_date = extract_range_data(smc_info, device_info, smcfb_info)
        p3 = plot_sleeping(df, min_date, max_date)
        st.pyplot(p3)

    elif smcfb_info in table_name + "_활동량":
        st.write("gg5")

    elif smcfb_info in table_name + "_분별HRV":
        st.write("GG6")

    # api call
    # api_call()


def save_plots_to_pdf_buffer(figs):
    pdf = FPDF()
    
    for fig in figs:
        img_stream = io.BytesIO()
        fig.savefig(img_stream, format="png")
        img_stream.seek(0)

        pdf.add_page()
        pdf.image(img_stream, x = 10, y = 20, w = 190)
    
    buffer = io.BytesIO()
    pdf.output(buffer, "F")
    buffer.seek(0)
    return buffer

def page_download():
    st.write("download")
    # 선택 하면 전체 plot 출력
    st.title("DataBase")
    # Add content for the about page
    with db.cursor() as cursor:
        cursor.execute("SELECT study_ID FROM fitbit_device_list")
        device_info_options = [row[0] for row in cursor.fetchall()]
    device_info = st.selectbox("SELECT Device Info", device_info_options)

    table_name = f"{device_info}"
    st.info(f"Selected Device Info: {device_info}")
    st.info(f"Table Name: {table_name}")

    st.write("this is table name: ", table_name)

    all_name = get_table_names(table_name)

    # st.write("this is all name: ", all_name)

    st.write(all_name)
    # AZM_data = get_table_data(all_name[0])
    # df, min_date, max_date = extract_range_data(AZM_data, device_info, all_name[0])

    # AZM_data = get_table_data(all_name[2])
    # df, min_date, max_date = extract_range_data(AZM_data, device_info, all_name[2])

    Sleep_data = get_table_data(all_name[4])
    df4, min_date4, max_date4 = extract_range_data(Sleep_data, device_info, all_name[4])
    psleep = plot_sleeping(df4, min_date4, max_date4)
    st.pyplot(psleep)

    Activity_data = get_table_data(all_name[5])
    df5, min_date5, max_date5 = extract_range_data(Activity_data, device_info, all_name[5])
    pact = plot_activity(df5, min_date5, max_date5)
    st.pyplot(pact)

    resting_heart_data = get_table_data(all_name[6])
    df6, min_date6, max_date6 = extract_range_data(resting_heart_data, device_info, all_name[6])
    p1 = plot_resting(df6, min_date6, max_date6)
    st.pyplot(p1)


    figs = [psleep, pact, p1]
    
    buffer = save_plots_to_pdf_buffer(figs)
    st.download_button(
        label="Download PDF",
        data=buffer,
        file_name=f"{device_info}.pdf",
        mime="application/pdf",
    )

# dbeaver 에서 해당되는 데이터 테이블 가져오기
def get_table_names(table_name):
    table_names = []
    # st.write("here is table name: ", table_name)
    table_name = table_name.replace('.', '_')
    try:
        with db.cursor() as cursor:
            query = "SHOW TABLES LIKE %s"
            like_pattern = f"%{table_name}%"  # LIKE 패턴 생성
            # st.write("Executing query: ", query % like_pattern)  # 쿼리 로그 출력
            cursor.execute(query, (like_pattern,))
            result = cursor.fetchall()
            for row in result:
                table_names.append(row[0])

            # st.write("this is table names: ", result)

    except pymysql.Error as e:
        st.error(f"An error occurred: {e}")  # Streamlit 에러 메시지 출력

    return table_names

# dbeaver 에서 해당하는 데이터 Columns 가져오기
def get_table_data(smcfb_info):
    data = []
    try:
        with db.cursor() as cursor:
            # Execute the query to retrieve the data from the table
            query = f"SELECT * FROM {smcfb_info}"
            data = pd.read_sql_query(query, db)
    
    except pymysql.Error as e:
        print(f"An error occurred: {e}")

    return data

db = pymysql.connect(host='119.67.109.156', 
                        port=3306,
                        user='root', 
                        password='Korea2022!', 
                        db='project_wd', 
                        charset='utf8')

# Usage example
def main():
    pages = {
        "환자데이터베이스": page_about,
        "전체 plot download": page_download
    }


    # 두번째 페이지에는 전체적으로 plot 하는 것을 보여주고 pdf 다운로드를 진행해볼예정
    
    st.sidebar.title("페이지이동")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Execute the selected page function
    pages[selection]()

if __name__ == '__main__':
    main()


