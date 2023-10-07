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
    plt.show()

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
    plt.show()


def run_plot(smcfb_info, smc_info, device_info):
#여기서 plot 을 표시해야함
    
    if smcfb_info in smcfb_info + "_휴식기심박수":
        df, min_date, max_date = extract_range_data(smc_info, device_info, smcfb_info)
        plot_resting(df, min_date, max_date)

    elif smcfb_info in smcfb_info + "_활동량":
        df, min_date, max_date = extract_range_data(smc_info, device_info, smcfb_info)
        plot_activity(df, min_date, max_date)

    elif smcfb_info in smcfb_info + "_AZM분별활동":
        st.write("gg1")

    elif smcfb_info in smcfb_info + "_분별심박수":
        st.write("gg2")

    elif smcfb_info in smcfb_info + "_수면상세":
        st.write("gg3")

    elif smcfb_info in smcfb_info + "_수면요약":
        st.write("gg4")

    elif smcfb_info in smcfb_info + "_활동량":
        st.write("gg5")

    elif smcfb_info in smcfb_info + "_분별HRV":
        st.write("GG6")


# 두번째 페이지
def page_about():
    st.title("DataBase")
    # Add content for the about page
    with db.cursor() as cursor:
        cursor.execute("SELECT study_ID FROM device_info_temp")
        device_info_options = [row[0] for row in cursor.fetchall()]

    device_info = st.selectbox("SELECT Device Info", device_info_options)
    st.write("This is ")

    st.write(device_info)

    table_name = f"{device_info}"
    st.info(f"Selected Device Info: {device_info}")
    st.info(f"Table Name: {table_name}")
    all_name = get_table_names(table_name)
    

    smcfb_info = st.selectbox("SELECT SMCFB Info", all_name)
    # st.write(smcfb_info)
    smc_info = get_table_data(smcfb_info)
    st.write(smc_info)

    run_plot(table_name, smc_info, device_info)




    
    if st.button("Run fitbit_auto.py"):
        subprocess.run(["python", "fitbit_auto.py"])
        st.success("fitbit_auto.py executed!")

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
        # "기기정보추가": page_home,
        "환자데이터베이스": page_about
    }
    
    st.sidebar.title("페이지이동")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Execute the selected page function
    pages[selection]()

if __name__ == '__main__':
    main()


