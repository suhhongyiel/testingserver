
import time
import requests
import pymysql

def db_insert_activity_data(user_id, dateTime, distance, steps, floors, calories):
    # DB 연결
    db = pymysql.connect(host='YOUR_DB_HOST', 
                         port=YOUR_DB_PORT,
                         user='YOUR_DB_USER', 
                         password='YOUR_DB_PASSWORD', 
                         db='YOUR_DB_NAME', 
                         charset='utf8')
    
    try:
        with db.cursor() as cursor:
            insert_query = '''
            INSERT INTO your_table_name (user_id, date, distance, steps, calories, floors)
            VALUES (%s, %s, %s, %s, %s, %s)
            '''
            cursor.execute(insert_query, (user_id, dateTime, distance, steps, calories, floors))
            db.commit()
    finally:
        db.close()

def Activity(set_time, user_id, header):
    try:
        # ... [API 요청 및 데이터 추출 코드]

        # 이 부분에서 DB에 데이터를 직접 삽입합니다.
        db_insert_activity_data(user_id, dateTime, distance, steps, floors, calories)
    except:
        print("Activity data is missing on " + set_time)

# ... [다른 함수 및 코드]

if __name__ == "__main__":
    # ... [메인 실행 코드]
