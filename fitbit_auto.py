import time
from pandas.tseries.offsets import MonthEnd
import pandas as pd 
import csv
import os
import requests
import numpy as np

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def Activity(set_time, user_id, directory, header):
    try:
        # 하루 활동량# 활동량
        # Step
        Activity_time_steps = requests.get(f'https://api.fitbit.com/1/user/-/activities/steps/date/'+set_time+'/1d.json', headers=header).json()
        # floors
        Activity_time_floors = requests.get(f'https://api.fitbit.com/1/user/-/activities/floors/date/'+set_time+'/1d.json', headers=header).json()
        # distance
        Activity_time_distance = requests.get(f'https://api.fitbit.com/1/user/-/activities/distance/date/'+set_time+'/1d.json', headers=header).json()
        # calories
        Activity_time_calories = requests.get(f'https://api.fitbit.com/1/user/-/activities/calories/date/'+set_time+'/1d.json', headers=header).json()
        # print("pass active")
    except:
        Activity_time_steps = -1
        Activity_time_floors = -1
        Activity_time_distance = -1
        Activity_time_calories = -1
        print("Activity data is missing on " + set_time)
    with open(directory + '/' + user_id + '_활동량.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            field = ["user_id", "date", "distance", "steps", "calories", "floors"]
            writer.writerow(field)
        
        # 하루 기준으로 code generated
        dateTime = Activity_time_steps['activities-steps'][0]['dateTime']
        distance = Activity_time_distance['activities-distance'][0]['value']
        steps = Activity_time_steps['activities-steps'][0]['value']
        # v1 code insert floors - calories
        floors = Activity_time_floors['activities-floors'][0]['value']
        calories = Activity_time_calories['activities-calories'][0]['value']

        # make csv
        writer.writerow([user_id, dateTime, distance, steps, calories, floors])


def AZM(set_time, user_id, directory, header):

    # AZM (분별 활동 interval) # cardioActive, fatBurnActive ...
    # activity_zone_minute = requests.get(f'https://api.fitbit.com/1/user/-/activities/active-zone-minutes/date/{year}-{month}-{day}/{year}-{month}-{yesterday}.json', headers=header).json()
    activity_zone_minute = requests.get(f'https://api.fitbit.com/1/user/-/activities/active-zone-minutes/date/'+set_time+'/1d.json', headers=header).json()
    if activity_zone_minute["activities-active-zone-minutes"] == []:
        print(f"AZM is not detected at {set_time}")
        
    # print("pass activity_zone_minute")
    # AZM (분별 활동 interval) # cardioActive, fatBurnActive ...
    try:
        with open(directory + '/' + user_id + '_AZM분별활동.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                field = ["user_id", "date", "activeZoneMinutes", "fatBurnActiveZoneMinutes", "cardioActiveZoneMinutes", ]
                writer.writerow(field)

            if activity_zone_minute["activities-active-zone-minutes"] == []:
                print("confirm")
                activeZoneMinutes = 0
                fatBurnActiveZoneMinutes = 0
                cardioActiveZoneMinutes = 0
                writer.writerow([user_id, set_time, activeZoneMinutes, fatBurnActiveZoneMinutes, cardioActiveZoneMinutes])
            else:

                for i in activity_zone_minute['activities-active-zone-minutes']:
                    
                    try:
                        activeZoneMinutes = i['value']['activeZoneMinutes']
                    except:
                        activeZoneMinutes = 0
                    try:
                        fatBurnActiveZoneMinutes = i['value']['fatBurnActiveZoneMinutes']
                    except:
                        fatBurnActiveZoneMinutes = 0
                    try:
                        cardioActiveZoneMinutes = i['value']['cardioActiveZoneMinutes']
                    except:
                        cardioActiveZoneMinutes = 0

                    writer.writerow([user_id, i['dateTime'], activeZoneMinutes, fatBurnActiveZoneMinutes, cardioActiveZoneMinutes])
    except:
        print("check try except")

def rest_HR(set_time, user_id, directory, header):
    # 휴식기 심박수
    try:
        heart_rate_request = requests.get(f'https://api.fitbit.com/1/user/-/activities/heart/date/'+set_time+'/'+set_time+'.json',
                                            headers=header).json()
        # print("pass heart_rate_request")
    except:
        print("would be 429 Error occur")

    with open(directory + '/' + user_id + '_휴식기심박수.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        
        if file.tell() == 0:
            field = ["user_id", "date","resting_hr"]
            writer.writerow(field)
        
        try:
            for i in heart_rate_request['activities-heart']:
                try:
                    rh_date = i['dateTime']
                    print(rh_date)
                    rh = i['value']['restingHeartRate']
                except:
                    print("resting heart rate is not detected")
                    rh = -1
                writer.writerow([user_id, rh_date, rh])
        except:
            print("would be Error when you request too much times api in short-times")

def HRV_min(set_time, user_id, directory,header ):
    # 분별 HRV
    HRV_min = requests.get(f'https://api.fitbit.com/1/user/-/hrv/date/'+set_time+'/all.json', headers=header).json()

    with open(directory + '/' + user_id + '_분별HRV.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            field = ["user_id", "date","time", "rmssd", "coverage", "hf", "lf" ]
            writer.writerow(field)
        try:
            for i in range(len(HRV_min['hrv'][0]['minutes'][0]['minute'])):
                try:
                    time_interval = HRV_min['hrv'][0]['minutes'][i]['minute'].split('T')
                    
                    writer.writerow([user_id, time_interval[0], time_interval[1], HRV_min['hrv'][0]['minutes'][i]['value']['rmssd'], HRV_min['hrv'][0]['minutes'][i]['value']['coverage'], HRV_min['hrv'][0]['minutes'][i]['value']['hf'], HRV_min['hrv'][0]['minutes'][i]['value']['lf']])
                except:
                    print("Error")
        except:
            none_value = -1
            writer.writerow([user_id, set_time, none_value, none_value, none_value, none_value, none_value])
            print("Patient does not have Deep sleep level")


def sleep_summary(set_time, user_id, directory, header):
    # 수면 요약
    # 수면 상태가 True 일때만 작동 아니면 csv 파일을 만들지 않음
    sleep_data = requests.get(f'https://api.fitbit.com/1.2/user/-/sleep/date/'+set_time+'.json', headers=header).json()

    with open(directory + '/' + user_id + '_수면요약.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            field = ["user_id", "date", "totalMinutesAsleep", "totalSleepRecords", "totalTimeInBed", "stages_deep", "stages_light", "stages_rem", "stages_wake", "efficiency", "cnt_deep", "cnt_light", "cnt_rem", "cnt_wake"]
            writer.writerow(field)
        try:
            sleep_dateTime = sleep_data['sleep'][0]['dateOfSleep']
        except:
            print("None Sleep Data, "+set_time+"this day")
            sleep_dateTime = set_time
        try:
            stages_deep = sleep_data['summary']['stages']['deep']
        except:
            stages_deep = -1
        try:
            stages_light = sleep_data['summary']['stages']['light']
        except:
            stages_light = -1
        try:
            stages_rem = sleep_data['summary']['stages']['rem']
        except:
            stages_rem = -1
        try:
            stages_wake = sleep_data['summary']['stages']['wake']
        except:
            stages_wake = -1
        
        #v1
        try:
            efficiency = sleep_data['sleep'][0]['efficiency']
        except:
            efficiency = -1

        try:
            cnt_deep = sleep_data['sleep'][0]['levels']['summary']['deep']['count']
        except:
            cnt_deep = -1
        
        try:
            cnt_light = sleep_data['sleep'][0]['levels']['summary']['light']['count']
        except:
            cnt_light = -1

        try:
            cnt_rem = sleep_data['sleep'][0]['levels']['summary']['rem']['count']
        except:
            cnt_rem = -1
        
        try:
            cnt_wake = sleep_data['sleep'][0]['levels']['summary']['wake']['count']
        except:
            cnt_wake = -1

        totalMinutesAsleep = sleep_data['summary']['totalMinutesAsleep']
        totalSleepRecords = sleep_data['summary']['totalSleepRecords']
        totalTimeInBed = sleep_data['summary']['totalTimeInBed']

        try:
            a = []
            for i in sleep_data['sleep']:
                a.append(i['efficiency'])

            sleep_efficiency = np.floor(np.mean(a))
        except:
            print("efficiency was not found")
        

        writer.writerow([user_id, sleep_dateTime, totalMinutesAsleep, totalSleepRecords, totalTimeInBed, stages_deep, stages_light, stages_rem, stages_wake, efficiency, cnt_deep, cnt_light, cnt_rem, cnt_wake ])

def sleep_dtl(set_time, user_id, directory, header):
    # 수면 상세
    # sleep_detail = requests.get(f'https://api.fitbit.com/1.2/user/-/sleep/list.json?afterDate='+set_time+'&sort=asc&offset=0&limit=1', headers=header).json()
    sleep_detail = requests.get(f'https://api.fitbit.com/1.2/user/-/sleep/date/' + set_time + '.json', headers=header).json()


    with open(directory + '/' + user_id + '_수면상세.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            field = ["user_id", "date","time_stamp", "sleep_stage", "sleep_duration"]
            writer.writerow(field)
        if sleep_detail['sleep'] == []:
            print("User Warning Code: Sleep data is empty , sent message to wear device")
            writer.writerow([user_id, set_time, '-1', '-1', '-1'])
        else:
            for i in sleep_detail['sleep']:
                for j in i['levels']['data']:
                    # sleep date time stamp
                    sleep_dateTime = j['dateTime']
                    sdTime = sleep_dateTime.split('T')


                    sleep_stage = j['level']
                    sleep_duration = j['seconds']

                    writer.writerow([user_id, i['dateOfSleep'], sdTime[1], sleep_stage, sleep_duration])

def min_by_heartrate(set_time, user_id, directory, header):
    # 분별 심박수
    # 만약 데이터가없으면 기본값을 배출하게 됩니다.
    Heart_rate_min = requests.get(f'https://api.fitbit.com/1/user/-/activities/heart/date/'+set_time+'/1d.json', headers=header).json()
    # print("pass Heart_rate_min (IF request too many then get error)")
    # 분별 심박수
    with open(directory + '/' + user_id + '_분별심박수.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            field = ["user_id", "date","time_min", "value"]
            writer.writerow(field)
    # By date
        if Heart_rate_min['activities-heart-intraday']['dataset'] == []:
            print("Except Code: Heart rate value in minute are empty")
            writer.writerow([user_id, Heart_rate_min['activities-heart'][0]['dateTime'], '-1', '-1'])
        else:
            for i in Heart_rate_min['activities-heart-intraday']['dataset']: #by daily 
                writer.writerow([user_id, Heart_rate_min['activities-heart'][0]['dateTime'], i['time'], i['value']])


    # Sig for top row
    row_sig = 1

def min_by_heartrate(set_time, user_id, directory, header):
    # 분별 심박수
    # 만약 데이터가없으면 기본값을 배출하게 됩니다.
    Heart_rate_min = requests.get(f'https://api.fitbit.com/1/user/-/activities/heart/date/'+set_time+'/1d.json', headers=header).json()
    # print("pass Heart_rate_min (IF request too many then get error)")
    # 분별 심박수
    with open(directory + '/' + user_id + '_분별심박수.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            field = ["user_id", "date","time_min", "value"]
            writer.writerow(field)
    # By date
        if Heart_rate_min['activities-heart-intraday']['dataset'] == []:
            print("Except Code: Heart rate value in minute are empty")
            writer.writerow([user_id, Heart_rate_min['activities-heart'][0]['dateTime'], '-1', '-1'])
        else:
            for i in Heart_rate_min['activities-heart-intraday']['dataset']: #by daily 
                writer.writerow([user_id, Heart_rate_min['activities-heart'][0]['dateTime'], i['time'], i['value']])


    # Sig for top row
    row_sig = 1


import re
import pandas as pd
import os
from datetime import datetime, timedelta

def get_existing_data_dates(directory, user_id):
    filepath = f'{directory}/{user_id}_수면요약.csv'
    if not os.path.exists(filepath):
        return []
    
    print("수면요약 getting!: ", user_id)

    data = pd.read_csv(filepath)
    return data['date'].tolist()

def get_missing_dates(directory, user_id, start_date, end_date):
    existing_dates = get_existing_data_dates(directory, user_id)

    formatted_existing_dates = [
    date if isinstance(date, str) and re.match(r'\d{4}-\d{2}-\d{2}', date) else datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') 
    for date in existing_dates if isinstance(date, str)
    ]

    all_dates = [start_date + timedelta(days=x) for x in range(0, (end_date - start_date).days)]
    
    missing_dates = [date for date in all_dates if date.strftime('%Y-%m-%d') not in formatted_existing_dates]
    
    print("this is missing_dates: ", missing_dates)

    return missing_dates

def get_data_for_user(user_id, missing_dates, header, directory):
    call_count = 0

    for current_date in missing_dates:
        if call_count >= 14:
            return False
        try:
            time_set = current_date.strftime('%Y-%m-%d')
            Activity(time_set, user_id, directory, header)
            AZM(time_set, user_id, directory, header)
            rest_HR(time_set, user_id, directory, header)
            HRV_min(time_set, user_id, directory, header)
            sleep_summary(time_set, user_id, directory,header)
            sleep_dtl(time_set, user_id, directory,header)
            min_by_heartrate(time_set, user_id, directory, header)
        except Exception as e:
            error_message = str(e)
            with open("error_log.txt", "a") as file:  # 'a'는 append 모드를 의미합니다.
                file.write(error_message + "\n")
                print("eception involved")
            

            call_count += 1

    return True
    