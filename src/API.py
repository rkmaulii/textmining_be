from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import lru_cache
import http.client
import joblib
import json
import os
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import sys
import time
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
from preprocess import preprocess
project_root = os.path.abspath(os.path.join(current_dir, ".."))
models_dir = os.path.join(project_root, "models")


app = Flask(__name__)
CORS(app) 

# =======================
# 1. Ambil Tweet by Username
# =======================

def get_user_rest_id(username, rapidapi_key):
    conn = http.client.HTTPSConnection("twitter241.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': rapidapi_key,
        'x-rapidapi-host': "twitter241.p.rapidapi.com"
    }
    endpoint = f"/user?username={username}"
    conn.request("GET", endpoint, headers=headers)
    res = conn.getresponse()
    data = res.read()
    try:
        json_data = json.loads(data.decode("utf-8"))
        user_result = (json_data.get("result", {})
                 .get("data", {})
                 .get("user", {})
                 .get("result", {})
                )
        rest_id = user_result.get("rest_id")
        if not rest_id:  
            print("⚠️ rest_id tidak ditemukan dalam response JSON")
            return None
        protected = user_result.get("privacy", {}).get("protected", False)
        if protected:
            return "PRIVATE"
        return rest_id
    except Exception as e:
        print("❌ Gagal mengambil rest_id:", e)
        return "Kesalahan"


def get_user_tweets(rest_id, count, rapidapi_key):
    conn = http.client.HTTPSConnection("twitter241.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': rapidapi_key,
        'x-rapidapi-host': "twitter241.p.rapidapi.com"
    }
    endpoint = f"/user-tweets?user={rest_id}&count={count}"
    conn.request("GET", endpoint, headers=headers)
    res = conn.getresponse()
    data = res.read()
    tweets_data = []
    try:
        json_data = json.loads(data.decode("utf-8"))
        instructions = json_data.get("result", {}).get("timeline", {}).get("instructions", [])
        for instruction in instructions:
            if instruction.get("type") == "TimelineAddEntries":
                for entry in instruction.get("entries", []):
                    if entry.get("entryId", "").startswith("tweet-"):
                        content = entry.get("content", {})
                        item_content = content.get("itemContent", {})
                        if not item_content:
                            continue
                        tweet_results = item_content.get("tweet_results", {}).get("result", {})
                        legacy = tweet_results.get("legacy", {})
                        lang = legacy.get("lang", "und")  
                        if legacy and not legacy.get("retweeted", False) and lang == "in":
                            tweets_data.append({
                                "text": legacy.get("full_text", ""),
                                "created_at": legacy.get("created_at", "N/A")
                            })
        if not tweets_data:
            return "zero"  
        return tweets_data
    except Exception as e:
        print("❌ Gagal parsing tweet:", e)
        return "Kesalahan_tweet"
    
# =======================
# 3. Prediksi
# =======================
def predict_texts(texts):
    model_path = os.path.join(models_dir, "svm_model9010.pkl")
    vectorizer_path = os.path.join(models_dir, "nc_tfidf_vectorizer.pkl")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    X = vectorizer.transform(texts)
    y_pred = model.predict(X)

    return y_pred

# =======================
# 4. Main Program
# =======================
try:
    from dotenv import load_dotenv

    print("Loading .env file")
    load_dotenv()
    print("Loaded .env file\n")
except Exception as e:
    print(f"Error loading .env file: {e}")

@app.route("/detect", methods=["POST"])
def main():
    data = request.json
    username = data.get("username","").strip()
    count = 20
    rapidapi_key=os.getenv("RAPID_KEY")
    if not rapidapi_key:
        return jsonify({"error": "API key tidak ditemukan di environment variables!"}), 500
    if not username:  # cek kosong
        return jsonify({"error": "Username tidak boleh kosong!"}), 400

    if " " in username:  # cek spasi di tengah
        return jsonify({"error": "Username tidak boleh mengandung spasi!"}), 400

    rest_id = get_user_rest_id(username, rapidapi_key)
    if rest_id is None:
        return jsonify({"error": "Akun tidak ditemukan."}), 404
    elif rest_id == "PRIVATE":
        return jsonify({"error": "Akun bersifat private."}), 403
    elif rest_id == "Kesalahan":
        return jsonify({"error": "Gagal mengambil rest_id (mungkin API error atau rate limit)."}), 500
    
    tweets = get_user_tweets(rest_id, count, rapidapi_key)
    if tweets == "zero" :
        return jsonify({"error": "Postingan kosong atau tidak berbahasa Indonesia."}), 404
    if tweets == "Kesalahan_tweet":
        return jsonify({"error": "Gagal mengambil tweets (mungkin API error atau rate limit)."}), 502
    
    # Tambahkan kolom clean_content
    tqdm.pandas()
    df = pd.DataFrame(tweets)
    df['clean_content'] = df['text'].astype(str).progress_apply(lambda x: preprocess(x, custom_norm=True))

    # Prediksi
    predictions = predict_texts(df['clean_content'].tolist())
    df['prediction'] = predictions

    # Untuk hasil
    hasil = []
    for _, row in df.iterrows():
            # parse datetime asli (pastikan sesuai format data kamu)
        try:
            dt_obj = datetime.strptime(row["created_at"], "%a %b %d %H:%M:%S %z %Y")  
            created_at_fmt = dt_obj.strftime("%d-%m-%Y %H:%M")
        except Exception:
            created_at_fmt = row["created_at"]  # fallback kalau parsing gagal

        hasil.append({
            "created_at": row["created_at"],
            "time": created_at_fmt,
            "text": row["text"],
            "clean_content": row["clean_content"],
            "prediction": row["prediction"]
    })    
   
    return jsonify (hasil)

if __name__ == "__main__":
    app.run(debug=True)