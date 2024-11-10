import schedule
import time
import requests

def send_file_to_discord(file_path, webhook_url):
    """Discordウェブフックを使ってファイルを送信する関数"""
    try:
        # ファイルを開いて読み込む
        with open(file_path, 'rb') as file:
            # ファイル内容をmultipart/form-dataで送信するための準備
            file_content = {'file': (file_path, file, 'application/octet-stream')}
            # DiscordのウェブフックURLにPOSTリクエストを送信
            response = requests.post(webhook_url, files=file_content)
            print(f"File sent successfully: {response.status_code}")
    except Exception as e:
        print(f"Failed to send file: {e}")

# ウェブフックURLとファイルパスの設定
webhook_url = 'https://discord.com/api/webhooks/example'  # 実際のウェブフックURLに置き換えてください
file_path = '/home/example/server/ChatGPT Web App（新Googleサイト用） - 履歴.csv'

# 毎日午後11時55分にファイル送信関数をスケジュール
schedule.every().day.at("23:55").do(send_file_to_discord, file_path=file_path, webhook_url=webhook_url)

# メインループ
while True:
    schedule.run_pending()
    time.sleep(1)
