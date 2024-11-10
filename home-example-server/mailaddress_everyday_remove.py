import datetime
from datetime import timedelta
import schedule
import time

def remove_lines_from_csv(original_file_path, output_file_path):
    try:
        # Open the file in write mode to clear its content
        with open(file_path, 'w', encoding='utf-8') as file:
            pass  # Writing nothing to the file will clear it
        
        print(f"All data including header has been cleared from: {file_path}")
    except Exception as e:
        print(f"Error: {e}")

# 同じファイルパスで上書きする例
file_path = '/home/example/server/ChatGPT Web App（新Googleサイト用） - メールアドレス記録（毎日リセット）.csv'
schedule.every().day.at("00:00").do(remove_lines_from_csv, original_file_path=file_path, output_file_path=file_path)

# メインループ
while True:
    schedule.run_pending()
    time.sleep(1)
