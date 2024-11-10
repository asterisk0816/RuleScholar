# 実験用のFastAPI起動コマンド（cdコマンドでパスに移動してから）：uvicorn FastAPI:app --reload --host 0.0.0.0 --port 8000
# ポート開放忘れずに！！！（8000番と80番）

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, validator
from typing import List
import csv
import datetime
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from rank_bm25 import BM25Okapi
from janome.tokenizer import Tokenizer
import pandas as pd
import torch.nn.functional as F
import re
import openai
import httpx
import asyncio
import unicodedata

import aioredis
import random
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from random import randint
from fastapi import HTTPException, Depends, FastAPI
from pydantic import BaseModel
from email.mime.multipart import MIMEMultipart

# radisサーバーを起動
"""
sudo apt update
sudo apt install redis-server

"""
redis = aioredis.from_url("redis://127.0.0.1:6379", encoding="utf-8", decode_responses=True)

# Discordウェブフックを使ったログ記録
import requests
webhook_url = 'https://discord.com/api/webhooks/example'
webhook_url2 = 'https://discord.com/api/webhooks/example2'


app = FastAPI()

# CORSを許可するオリジンのリスト
origins = [
    "http://172.16.9.200", # フロントエンドのIPアドレス（8000ポート開放忘れずに）
    "http://localhost:3000",  # Reactアプリなどのフロントエンドのオリジン
    "http://localhost:8000",  # 自身のオリジンも追加できます
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 許可するオリジン
    allow_credentials=True,
    allow_methods=["*"],  # 許可するHTTPメソッド
    allow_headers=["*"],  # 許可するHTTPヘッダー
)



class EmailValidationRequest(BaseModel):
    email: str
    password: str

    # メールアドレスにドメインを追加
    @validator('email', pre=True, always=True)
    def append_domain(cls, v):
        return f"{v}@example.com"

    # メールアドレス範囲検証
    @validator('email')
    def validate_email_range(cls, v):
        user_id = int(v.split('@')[0])
        if 32181001 <= user_id <= 32246999:
            return v
        # 管理者例外
        elif user_id == 00000000:
            return v
        
        # 送信するデータ
        data = {
            'content': f'【ValueError】Email is out of the allowed range {user_id}@example.com',
        }
        # POSTリクエストを送信
        response = requests.post(webhook_url, json=data)

        raise ValueError('Email is out of the allowed range')


class ChatGPTRequest(BaseModel):
    loggedInEmail: str
    message: str


class EmailValidationRequestForReset(BaseModel):
    email: str
    password: str

    # メールアドレスにドメインを追加
    @validator('email', pre=True, always=True)
    def append_domain(cls, v):
        return f"{v}@example.com"

    # メールアドレス範囲検証
    @validator('email')
    def validate_email_range(cls, v):
        user_id = int(v.split('@')[0])
        if 32181001 <= user_id <= 32246999:
            return v
        # 管理者例外
        elif user_id == 00000000:
            return v
        
        # 送信するデータ
        data = {
            'content': f'【ValueError】Email is out of the allowed range {user_id}@example.com',
        }
        # POSTリクエストを送信
        response = requests.post(webhook_url, json=data)

        raise ValueError('Email is out of the allowed range')
    

class EmailValidationRequestForChange(BaseModel):
    email: str
    enterednewpassword: str
    enteredSixPassword: str

    # メールアドレスにドメインを追加
    @validator('email', pre=True, always=True)
    def append_domain(cls, v):
        return f"{v}@example.com"
    



# リクエスト時間確認
def request_time_validation():
    current_hour = datetime.datetime.now().hour
    if not 8 <= current_hour <= 16:

        # 送信するデータ
        data = {
            'content': '【400】Request out of allowed time',
        }
        # POSTリクエストを送信
        response = requests.post(webhook_url, json=data)

        raise HTTPException(status_code=400, detail="Request out of allowed time")
    return True

# 特定行のデータ存在チェック
def check_data_in_specific_line():
    with open("/home/example/server/ChatGPT Web App（新Googleサイト用） - アクセス制限検出.csv", mode='r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader, start=1):
            if i == 51 and row:
                
                # 送信するデータ
                data = {
                    'content': '【400】Access restricted due to specific data presence',
                }
                # POSTリクエストを送信
                response = requests.post(webhook_url, json=data)

                raise HTTPException(status_code=400, detail="Access restricted due to specific data presence")
    return True

# ファイルパスの存在を確認する関数
async def validate_file_path(file_path: str) -> bool:
    if file_path is None:

        # 送信するデータ
        data = {
            'content': '【400】File path is required',
        }
        # POSTリクエストを送信
        response = requests.post(webhook_url, json=data)

        raise HTTPException(status_code=400, detail="File path is required")
    if not os.path.exists(file_path):

        # 送信するデータ
        data = {
            'content': '【404】File not found',
        }
        # POSTリクエストを送信
        response = requests.post(webhook_url, json=data)

        raise HTTPException(status_code=404, detail="File not found")
    return True

# メールアドレスとパスワードの検証
async def validate_email_password(email: str, password: str):
    user_id = int(email.split('@')[0])
    year = email[2:4]  # 3番目と4番目の文字

    print(email)
    print(year)

    file_path = f"/home/example/server/login_data/ChatGPT Web App（新Googleサイト用） - ログイン情報（{year}）.csv"

    # 管理者認証
    if email == "00000000@example.com":
        if password == "00000000":
            return True
        else:
            
            # 送信するデータ
            data = {
                'content': f'【401】Invalid email/password combination {email} {password}',
            }
            # POSTリクエストを送信
            response = requests.post(webhook_url, json=data)

            raise HTTPException(status_code=401, detail="Invalid email/password combination")

    # 対応するCSVファイルが存在するか確認
    if not os.path.exists(file_path):

        # 送信するデータ
        data = {
            'content': '【404】Login data file not found',
        }
        # POSTリクエストを送信
        response = requests.post(webhook_url, json=data)

        raise HTTPException(status_code=404, detail="Login data file not found")
    
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        first_user_id = int(f"32{year}1001")
        row_number = user_id - first_user_id
        for i, row in enumerate(reader):
            if i == row_number:
                _, row_password = row[0].split('（')
                row_password = row_password[:-1]  # 末尾の）を削除
                if row_password == "********":
                    # 送信するデータ
                    data = {
                        'content': f'【401】Delieted account by Admin {email}',
                    }
                    # POSTリクエストを送信
                    response = requests.post(webhook_url, json=data)
                    raise HTTPException(status_code=401, detail="無効なアカウント")
                if row_password == password:
                    return {"authenticated": True}  # ユーザー認証成功
                break

    # 送信するデータ
    data = {
        'content': f'【401】Invalid email/password combination {email} {password}',
    }
    # POSTリクエストを送信
    response = requests.post(webhook_url, json=data)

    raise HTTPException(status_code=401, detail="Invalid email/password combination")

# メールアドレスの一意性を確認（2列目にメールアドレスを含む）
async def validate_email_uniqueness(email: str):
    file_path = "/home/example/server/ChatGPT Web App（新Googleサイト用） - メールアドレス記録（毎日リセット）.csv"
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            # 2列目を対象に検索
            if row and len(row) > 1 and row[1] == email:

                # 送信するデータ
                data = {
                    'content': f'【409】Email already exists {email}',
                }
                # POSTリクエストを送信
                response = requests.post(webhook_url, json=data)

                raise HTTPException(status_code=409, detail="Email already exists")
    return True

# タイムスタンプとメールアドレス記録
async def record_email(email: str):
    # 管理者アカウントは記録しない
    if email.split('@')[0] == "00000000":
        return True

    file_paths = [
        "/home/example/server/ChatGPT Web App（新Googleサイト用） - アクセス制限検出.csv",
        "/home/example/server/ChatGPT Web App（新Googleサイト用） - メールアドレス記録（毎日リセット）.csv"
    ]

    for file_path in file_paths:
        with open(file_path, mode='a', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # 現在の日時を取得
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # タイムスタンプとメールアドレスを書き込む
            writer.writerow([current_time, email])




# 不適切な言葉のリスト
inappropriate_keywords: List[str] = [
    "不適切な単語1",
    "不適切な単語2",
    # 他にも不適切な単語を追加できます
]

def filter_message_content(message: str) -> str:
    """
    メッセージ内の不適切な言葉をフィルタリングします。
    不適切な内容が含まれている場合は、HTTPExceptionを投げます。
    """
    # 文字列を正規化して全角文字を半角文字に統一する
    def normalize_text(text: str) -> str:
        return unicodedata.normalize('NFKC', text).lower()

    # メッセージを正規化
    normalized_message = normalize_text(message)

    # キーワードリストも正規化
    normalized_keywords = [normalize_text(keyword) for keyword in inappropriate_keywords]

    # 不適切なキーワードの存在をチェック
    if any(re.search(keyword, normalized_message, flags=re.IGNORECASE) for keyword in normalized_keywords):
        raise HTTPException(status_code=400, detail="Message contains inappropriate content")

    print(f"\nフィルタリングにクリアしたクエリ：{message}")
    return message


########## ここから複雑な検索関数に入るため、余計な操作はしないようにすること ##########

# GPUの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# トークナイザーのパスを指定
tokenizer_path = '/home/example/server/stsb-xlm-r-multilingual'  # トークナイザーのパスを指定
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# モデルのパスを指定
model_path = '/home/example/server/classify-net（epoc-10000）'
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=11)
model.to(device)



def search_bm25(docs, query, num_results=3):
    # BM25による検索
    tokenizer = Tokenizer()
    # 修正: ドキュメントを文字列に変換
    tokenized_docs = [list(tokenizer.tokenize(str(doc), wakati=True)) for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = list(tokenizer.tokenize(str(query), wakati=True))  # queryも文字列に変換
    scores = bm25.get_scores(tokenized_query)
    top_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_results]
    return [(docs[i], scores[i]) for i in top_indexes]

def predict_category(text, threshold=0.5):  # 閾値は必要に応じて調整
    # テキストのカテゴリを予測
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        max_prob, predicted_category = torch.max(probabilities, dim=1)
        # 閾値を超える確信度がある場合のみカテゴリを返す
        if max_prob.item() > threshold:
            return predicted_category.item()
        else:
            return -1  # 確信度が閾値を下回る場合は-1を返す



def search_bm25_all_data(query, csv_file_paths, num_results=3):
    # トークナイザの準備
    tokenizer = Tokenizer()
    # 全データに対する検索結果を保持するリスト
    all_search_results = []
    # 各CSVファイルに対してBM25検索を実施
    for path in csv_file_paths.values():
        df = pd.read_csv(path, header=None)
        docs = df[0].tolist()
        tokenized_docs = [list(tokenizer.tokenize(str(doc), wakati=True)) for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = list(tokenizer.tokenize(str(query), wakati=True))
        scores = bm25.get_scores(tokenized_query)
        top_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_results]
        for i in top_indexes:
            all_search_results.append((docs[i], scores[i]))

    # 全データからの検索結果をスコア順にソート
    all_search_results.sort(key=lambda x: x[1], reverse=True)

    # 最上位の結果のみを返す
    return all_search_results[:num_results]



def generate_nouns_with_similarity(query):
    tokenizer = Tokenizer()
    # 形態素解析による名詞抽出
    nouns = [token.surface for token in tokenizer.tokenize(query) if token.part_of_speech.startswith('名詞')]
    return nouns

# 名詞に基づいて文書を検索しランキングする関数
def search_documents_by_nouns(csv_file_paths, query):
    # 類似語を考慮した名詞リストの生成
    nouns = generate_nouns_with_similarity(query)
    document_scores = {}

    for label, path in csv_file_paths.items():
        try:
            df = pd.read_csv(path, header=None)
            for index, row in df.iterrows():
                doc = str(row[0])
                # 文書内に含まれる各名詞（または類似語辞書で定義された語句）の出現回数をカウント
                score = sum(doc.count(noun) for noun in nouns)
                if score > 0:
                    # 文書とそのスコアを記録
                    document_scores[doc] = score
        except Exception as e:
            print(f"ファイル {path} の処理中にエラーが発生しました: {e}")

    # スコアに基づいて文書をランキング
    ranked_documents = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked_documents:
        return "該当する結果なし"
    else:
        # スコアが最も高い上位の文書を選択
        return ranked_documents


##########辞書型検索（関数郡）##########

# クエリに基づいて言い換えを検索する辞書（左側は自由形式、右側は単語や名詞推奨）
replacement_dict = {
    "Instagram": "SNS",
    "インスタグラム": "SNS",
    "インスタ": "SNS",
    "ツイッター": "SNS",
    "Twitter": "SNS",
    "X": "SNS",
    "エックス": "SNS",
    "facebook": "SNS",
    "FaceBook": "SNS",
    "フェイスブック": "SNS",
    "フェイス": "SNS",
    "Misskey": "SNS",
    "ミスキー": "SNS",
    "LINE": "SNS",
    "ライン": "SNS",
    "Tiktok": "SNS",
    "BeReal": "SNS",
    "Bereal": "SNS",
    "beReal": "SNS",
    "bereal": "SNS",
    "ビーリアル": "SNS",
    "Bluesky": "SNS",
    "BlueSky": "SNS",
    "bluesky": "SNS",
    "ブルースカイ": "SNS",
    "pinterest": "SNS",
    "Pinterest": "SNS",
    "Threads": "SNS",
    "threads": "SNS",
    "スレッズ": "SNS",
    "tik": "SNS",
    "Tik": "SNS",
    "Tok": "SNS",
    "tok": "SNS",
    "TikTok": "SNS",
    "tiktok": "SNS",
    "ティックトック": "SNS",
    "自販機": "自動販売機",
}

def normalize_string(s):
    normalized = unicodedata.normalize('NFKC', s).lower()
    return normalized

def search_replacement_in_csv_files(csv_file_paths, query):
    results = []
    query_normalized = normalize_string(query)

    for key, replacement in replacement_dict.items():
        if normalize_string(key) in query_normalized:
            replacement_normalized = normalize_string(replacement)
            for csv_key, path in csv_file_paths.items():
                if 0 <= csv_key <= 10:
                    df = pd.read_csv(path, header=None)
                    for index, row in df.iterrows():
                        if replacement_normalized in normalize_string(str(row[0])):
                            results.append(row[0])
    return results

##########辞書型検索終了##########


def main(query):

    # 結果を格納するためのリストを初期化
    results_list = []

    results_list.append(f"\nクエリ: {query}")

    predicted_category = predict_category(query, threshold=0.5)  # 閾値を適宜設定
    if predicted_category == -1:
      results_list.append("確信度が閾値を下回るため、カテゴリに基づく検索は行いません。")
    else:
      results_list.append(f"\nカテゴリ（Powered by BERT）: {predicted_category}")

    # CSVファイルのパスをカテゴリに基づいて選択
    csv_file_paths = {
        0: '/home/example/server/school_rules_corrected_corrected/rule（0）.csv',
        1: '/home/example/server/school_rules_corrected_corrected/rule（1）.csv',
        2: '/home/example/server/school_rules_corrected_corrected/rule（2）.csv',
        3: '/home/example/server/school_rules_corrected_corrected/rule（3）.csv',
        4: '/home/example/server/school_rules_corrected_corrected/rule（4）.csv',
        5: '/home/example/server/school_rules_corrected_corrected/rule（5）.csv',
        6: '/home/example/server/school_rules_corrected_corrected/rule（6）.csv',
        7: '/home/example/server/school_rules_corrected_corrected/rule（7）.csv',
        8: '/home/example/server/school_rules_corrected_corrected/rule（8）.csv',
        9: '/home/example/server/school_rules_corrected_corrected/rule（9）.csv',
        10: '/home/example/server/school_rules_corrected_corrected/rule（10）.csv'
        # 他のカテゴリに対応するCSVファイルのパスを追加
    }

    if predicted_category == -1:
      pass
    else:
      selected_csv_path = csv_file_paths[predicted_category]
      df = pd.read_csv(selected_csv_path, header=None)
      docs = df[0].tolist()  # 最初の列（インデックスは0）にテキストデータがあると仮定

      # BM25によるカテゴリ内の検索結果
      category_search_results = search_bm25(docs, query)
      results_list.append("\nカテゴリの検索結果（Powered by BM25）:")
      for doc, score in category_search_results:
          results_list.append(f"スコア: {score:.2f}, {doc}")
          
    # 新しいファイルパスの追加
    new_file_paths = [
        '/home/example/server/school_rules_additional/patch.csv',
    ]
    new_key = max(csv_file_paths.keys()) + 1  # 現在の最大キーの次の値
    for new_path in new_file_paths:
        if os.path.exists(new_path):
            csv_file_paths[new_key] = new_path
            new_key += 1
        else:
            results_list.append(f"ファイルが存在しないためスキップされました: {new_path}")

    # 全データに対するBM25検索結果
    all_data_search_results = search_bm25_all_data(query, csv_file_paths)
    results_list.append("\n全データ検索結果（Powered by BM25）:")
    for doc, score in all_data_search_results:
        results_list.append(f"スコア: {score:.2f}, {doc}")

    # 形態素解析による検索結果
    results_list.append("\n形態素解析による検索結果（Powered by Janome）:")
    janome_search_results = search_documents_by_nouns(csv_file_paths, query)
    if janome_search_results == "該当する結果なし":
        results_list.append(janome_search_results)
    else:
        # ここを修正: extract_nouns ではなく generate_nouns_with_similarity を使用
        extended_nouns = generate_nouns_with_similarity(query)
        results_list.append("抽出した名詞：" + ", ".join(extended_nouns))
        for doc, score in janome_search_results:
            results_list.append(f"スコア: {score}：{doc}")

    # 言い換え拡張検索結果
    results_list.append("\n言い換え拡張検索結果:")
    search_results = search_replacement_in_csv_files(csv_file_paths, query)
    if search_results:
        for result in search_results:
            results_list.append(result)
    else:
        results_list.append("クエリに対応する言い換えが辞書にありません。")
    
    return results_list

########## 検索関数終了 ##########


def get_chatgpt_response(query: str):

    # クエリがリスト形式の場合は、改行で結合して一つの文字列に変換
    if isinstance(query, list):
        query = "\n".join(query).strip()
    else:
        query = query

    print(f"\n{query}")
    
    # ChatGPT APIの呼び出し
    openai.api_key = "example"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # 使用するモデル
        messages=[
            {"role": "system", "content": "You are an excellent assistant who never answers questions about anything other than school rules. Below are questions about school rules and search results from a database. Please kindly respond to the query in Japanese with the most credible sentence that reflects the results of the query. Additionally, paraphrase expansion search is highly reliable information.When asked for a reason, please provide a specific reason. When asked how to do something, please provide specific instructions on how to do it. If you do not know, please answer that you do not know."},
            {"role": "user", "content": query}
        ],
        max_tokens=200  # 生成するトークンの最大数
    )

    try:
        # 応答テキストの取得
        text_response = response['choices'][0]['message']['content']
        print(f"\nChatGPTの回答：{text_response}")
        return text_response
    
    except KeyError:
        # 応答にテキストが含まれていない場合のエラーハンドリング
        print({"status": "error", "response": "Response does not contain 'text'."})

        # 送信するデータ
        data = {
            'content': """【KeyError】{"status": "error", "response": "Response does not contain 'text'."}""",
        }
        # POSTリクエストを送信
        response = requests.post(webhook_url, json=data)

        return({"status": "error", "response": "Response does not contain 'text'."})



########## ここから別のコード #########

# データの記録と取得
async def store_reset_code(email: str, code: str):
    # 600秒 = 10分
    await redis.set(email, code, ex=600)

async def get_reset_code(email: str):
    return await redis.get(email)


# メール送信機能
def send_email(from_email, to_email, subject, message, smtp_password):
    # MIMEMultipartオブジェクトを生成
    msg = MIMEMultipart('alternative')
    msg['From'] = Header(from_email, 'utf-8')
    msg['To'] = Header(to_email, 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')

    # HTMLバージョンを添付
    msg.attach(MIMEText(message, 'html', 'utf-8'))

    # サーバーに接続し、メールを送信
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(from_email, smtp_password)
    server.sendmail(from_email, [to_email], msg.as_string())
    server.quit()



# APIの定義
@app.post("/api/validateAndRecordEmail")
# 時間確認（request_time_validation関数）とアクセス制限検出（check_data_in_specific_line関数）をまず行うことで、管理者アカウントとのコードを統一させる

# async def validate_and_record_email(request: EmailValidationRequest, time_valid: bool = Depends(request_time_validation), data_check: bool = Depends(check_data_in_specific_line)):
async def validate_and_record_email(request: EmailValidationRequest, data_check: bool = Depends(check_data_in_specific_line)):

    # ファイルが存在するか確認（validate_file_path関数）
    await validate_file_path(file_path="/home/example/server/ChatGPT Web App（新Googleサイト用） - アクセス制限検出.csv")
    await validate_file_path(file_path="/home/example/server/ChatGPT Web App（新Googleサイト用） - メールアドレス記録（毎日リセット）.csv")

    # メールアドレスとパスワードの検索と照合（validate_email_password関数）、メールアドレスの重複を確認（validate_email_uniqueness関数）、メールアドレスの記録（record_email関数）を行う
    await validate_email_password(request.email, request.password)
    await validate_email_uniqueness(request.email)
    await record_email(request.email)

    return {"status": "ok", "email": request.email}



# ChatGPTバックエンド
@app.post("/api/chatGPTbackend")
# async def chat_gpt_backend(request: ChatGPTRequest, time_valid: bool = Depends(request_time_validation)):
async def chat_gpt_backend(request: ChatGPTRequest):

    # ユーザーメッセージのフィルタリング
    filtered_message = filter_message_content(request.message)

    # main関数を呼び出して処理を行う
    searched_query = main(filtered_message)

    # ChatGPTのAPIに送信し、応答を得る
    chatGPT_response = get_chatgpt_response(searched_query)

    # 応答をフィルタリング
    # chatGPT_responseが文字列であることを確認し、フィルタリング関数を適用する
    if isinstance(chatGPT_response, str):
        filtered_chatGPT_response = filter_message_content(chatGPT_response) 
    else:

        # 送信するデータ
        data = {
            'content': "【500】Invalid response from ChatGPT.",
        }
        # POSTリクエストを送信
        response = requests.post(webhook_url, json=data)
        
        # 応答が文字列でない場合はエラーメッセージを返す
        raise HTTPException(status_code=500, detail="Invalid response from ChatGPT.")

    # CSVファイルに記録する
    csv_file_path = "/home/example/server/ChatGPT Web App（新Googleサイト用） - 履歴.csv"
    with open(csv_file_path, mode='a', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # 現在のタイムスタンプを取得
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # CSVに書き込むデータの準備
        row = [current_time, request.loggedInEmail, request.message, filtered_chatGPT_response]
        # CSVに行を追加
        writer.writerow(row)

    # クエリがリスト形式の場合は、改行で結合して一つの文字列に変換
    if isinstance(searched_query, list):
        searched_query = "\n".join(searched_query).strip()
    else:
        searched_query = searched_query
    
    # ログメッセージを作成
    log_message = f"メールアドレス： {request.loggedInEmail}\n\nフィルタリング済みのクエリ：{filtered_message}\n{searched_query}\n\nChatGPTの初期回答：{chatGPT_response}\n\nChatGPTのフィルタリング済みの回答：{filtered_chatGPT_response}"

    # メッセージが2000文字を超える場合、2000文字に切り上げる
    if len(log_message) > 2000:
        log_message = log_message[:2000]

    # 送信するデータ
    data = {
        'content': log_message
    }

    # POSTリクエストを送信
    response = requests.post(webhook_url2, json=data)

    # ChatGPTの応答を返す
    return {"status": "ok", "response": filtered_chatGPT_response}



# ログイン＆6桁コード生成＆メール送信関数
@app.post("/api/validateAndRecordEmailForReset")

# async def validate_and_record_email_for_reset(request: EmailValidationRequestForReset, time_valid: bool = Depends(request_time_validation), data_check: bool = Depends(check_data_in_specific_line)):
async def validate_and_record_email_for_reset(request: EmailValidationRequestForReset, data_check: bool = Depends(check_data_in_specific_line)):

    if request.email == "00000000@example.com":
        raise HTTPException(status_code=403, detail="Can't Change")
    else:
        # メールアドレスとパスワードの検索と照合（ここでは省略）
        await validate_email_password(request.email, request.password)

    code = randint(100000, 999999)  # ランダムな6桁の数字を生成
    await store_reset_code(request.email, str(code))
    
    # HTMLメールの内容を設定
    from_email = 'noreply@example.com'
    to_email = request.email
    subject = f'【RuleScolar】2段階認証コードは {code} です'
    message = f"""
<html>
<head>
<style>
  body {{
    font-family: 'Arial', sans-serif;
    margin: 0;
    padding: 0;
    background-color: #ffffff;
  }}
  .container {{
    max-width: 600px;
    margin: 20px auto;
    padding: 20px;
    background-color: #f5f5f5;
    border: 1px solid #011627;
    border-radius: 50px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }}
  .header {{
    background-color: #f5f5f5; /* 背景色を薄いグレーに設定 */
    color: #011627; /* テキストの色を赤に設定 */
    padding: 10px;
    text-align: center;
    border-radius: 50px 50px 0 0;
    border-bottom: 1px solid #011627; /* 下線を赤色で2pxの太さに設定 */
  }}
  .code {{
    margin: 20px 0;
    padding: 10px;
    background-color: #ffffff;
    text-align: center;
    border-radius: 30px;
  }}
  .footer {{
    text-align: center;
    margin-top: 20px;
    font-size: 12px;
    color: #666;
  }}
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>R u l e S c h o l a r</h1>
    </div>
    <br><h3><strong>{request.email} 様</strong></h3>
    <p>アカウントのセキュリティコードは以下のとおりです。</p>
    <div class="code">
      <strong style="font-size: 22px; color: #F71735;">{code}</strong>
    </div>
    <p>コードは10分間有効です。<br>このメールに心当たりがない場合、またはご不明な点がある場合は、すぐにアカウントのセキュリティを確認してください。</p>
    <div class="footer">
      このメールは自動送信されていますので、こちらに返信しないでください。<br>
      ご質問がある場合は、生徒会までお問い合わせください。<br><br>
      ©生徒会
    </div>
  </div>
</body>
</html>
"""
    smtp_password = 'example'  # Googleのアプリパスワード
    send_email(from_email, to_email, subject, message, smtp_password)

    return {"status": "ok", "email": request.email}



# 6桁のコード検証と変更依頼
@app.post("/api/ChangePassword")

# async def change_password(request: EmailValidationRequestForChange, time_valid: bool = Depends(request_time_validation)):
async def change_password(request: EmailValidationRequestForChange):

    print(request.email, request.enterednewpassword, request.enteredSixPassword)

    # Redisからメールアドレスに関連するコードを取得
    stored_code = await get_reset_code(request.email)
    
    if stored_code is None:
        raise HTTPException(status_code=404, detail="No reset code found for this email address.")
    
    # 提供されたコードが保存されたコードと一致するか確認
    if stored_code != request.enteredSixPassword:
        raise HTTPException(status_code=403, detail="Invalid code provided.")
    
    # コードが一致した場合、成功をプリントする
    print("Verification successful.")

    # コードが一致した場合、CSVファイルにメールアドレスとパスワードを書き込む
    csv_path = "/home/example/server/ChatGPT Web App（新Googleサイト用） - パスワード自動変更.csv"  # CSVファイルのパスを設定
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([request.email, str(request.enterednewpassword)])

    return {"status": "success", "message": "Password changed successfully."}
