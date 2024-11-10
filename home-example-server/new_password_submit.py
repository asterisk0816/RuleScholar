import csv
import schedule
import time
import requests
import os
from datetime import datetime


def update_passwords(reference_file_path):
    try:
        with open(reference_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 2:
                    print(f"Skipping incomplete or empty row: {row}")
                    continue
                email, new_password = row
                year = email[2:4]
                target_file_path = f'/home/example/server/login_data/ChatGPT Web App（新Googleサイト用） - ログイン情報（{year}）.csv'
                if not update_password(target_file_path, email, new_password):
                    print(f"No update needed or error for {email} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        clear_file_content(reference_file_path)
    except Exception as e:
        print(f"Error processing file {reference_file_path}: {e}")

def clear_file_content(file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            pass
        print(f"Reference file content has been cleared: {file_path}")
    except Exception as e:
        print(f"Failed to clear file {file_path}: {e}")

def update_password(target_file_path, email, new_password):
    try:
        updated = False
        updated_lines = []
        with open(target_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if not row or len(row) < 1:
                    continue
                stored_email, old_password = row[0].split('（')
                old_password = old_password[:-1]
                updated_line = f'{stored_email}（{new_password}）' if stored_email == email else f'{stored_email}（{old_password}）'
                updated_lines.append(updated_line)
                if stored_email == email:
                    updated = True
        if updated:
            with open(target_file_path, mode='w', encoding='utf-8') as file:
                writer = csv.writer(file)
                for line in updated_lines:
                    writer.writerow([line])
            return True
        return False
    except Exception as e:
        print(f"Error updating email {email} in file {target_file_path}: {e}")
        return False

# 参照ファイルパス設定
reference_file_path = '/home/example/server/ChatGPT Web App（新Googleサイト用） - パスワード自動変更.csv'

# スケジュール設定
# schedule.every().day.at("23:55").do(send_file_to_discord, reference_file_path=reference_file_path)
schedule.every().day.at("23:55").do(update_passwords, reference_file_path=reference_file_path)

# メインループ
while True:
    schedule.run_pending()
    time.sleep(1)  # 毎秒チェック
