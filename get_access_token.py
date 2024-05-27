import os
# os.system("pip install -qU selenium==4.9.1 pandas")

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

import pandas as pd 


# install for current chrome driver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager



oauth_data = pd.read_excel("Fitbit_device_log.xlsx")
for index in range(0,len(oauth_data)):
    

    if pd.isna(oauth_data.ACCESS_TOKEN[index])==False:
        continue

    CLIENT_ID = oauth_data.CLIENT_ID[index]
    CLIENT_SECRET = oauth_data.CLIENT_SECRET[index]
    id = oauth_data.Google_id[index]
    pw = oauth_data.Google_pw[index]

    print("This is check for oauth_data.study_ID:: ", oauth_data.study_ID[index])
    print("This is check INDEX number: ", index)
    if CLIENT_ID == 'nan':
        print("Please, get client id from https://dev.fitbit.com/apps/new, and enter client_id and secret_key on Fitbit_device_log.xlsx")

    if id == 'nan':
        print("Please, enter your fitbit id and pw on Fitbit_device_log.xlsx")

    path = "C:/Python Library/chrome_driver/chromedriver"
    url = "https://www.fitbit.com/oauth2/authorize?response_type=token&client_id="+CLIENT_ID+"&redirect_uri=http://127.0.0.1:8080/&expires_in=31536000&scope=activity+nutrition+heartrate+location+nutrition+profile+settings+sleep+social+weight"

    # 크롬 옵션 정의 (1이 허용, 2가 차단)
    # chrome_options = Options()
    chrome_options = webdriver.ChromeOptions()
    prefs = {"profile.default_content_setting_values.notifications": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    
    # driver = webdriver.Chrome(path, options=chrome_options)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)

    time.sleep(5)
    # 드라이버가 Facebook에 접근했는지 title을 확인함.
    assert "Fitbit Accounts" in driver.title
    # 가끔 변경 됨 
    login_box = driver.find_element(By.ID, "ember592")
    login_box.send_keys(id)
    login_box = driver.find_element(By.ID, "ember593")
    login_box.send_keys(pw)
    login_box.send_keys(Keys.RETURN)

    time.sleep(10)
    ACCESS_TOKEN=driver.current_url.split('access_token=')[1].split('&user_id=')[0]
    oauth_data.ACCESS_TOKEN[index]=ACCESS_TOKEN

    print(ACCESS_TOKEN)
    print("Oauth data: " , oauth_data.ACCESS_TOKEN[index])

    # 창 닫기
    driver.close()

    oauth_data.to_excel("Fitbit_device_log.xlsx")

