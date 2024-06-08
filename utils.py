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