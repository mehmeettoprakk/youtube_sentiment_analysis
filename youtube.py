import os
import re
import nltk
import pandas as pd
import numpy as np
import tkinter as tk
from tokenizer import clean_text
from ysa_model import word2vec_model, model
from tkinter import ttk, messagebox
from googleapiclient.discovery import build
from dotenv import load_dotenv


# NLTK kütüphanesinden gerekli kaynakları indirme
nltk.download('punkt')
nltk.download('stopwords')

# YouTube API anahtarını alma işlemi
load_dotenv()
api_key = os.getenv('YOUTUBE_API_KEY')

# YouTube API'yi başlat
youtube = build('youtube', 'v3', developerKey=api_key)

def extract_video_id(url):
    # Video ID içeriyorsa
    if "v=" in url:
        video_id_match = re.search(r'v=([0-9A-Za-z_-]{11})', url)
        return video_id_match.group(1) if video_id_match else None
    # Video ID içermiyorsa
    video_id_match = re.search(r'/([0-9A-Za-z_-]{11})', url)
    return video_id_match.group(1) if video_id_match else None

# Yorumları çekmek için fonksiyon
def get_comments(video_id):
    try:
        comments_data = []
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=52
        )
        response = request.execute()

        for item in response['items']:
            comment_info = {
                'KanalId': item['snippet']['topLevelComment']['snippet']['authorChannelId'].get('value', ''),
                'Yorum Yazarı': item['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                'VideoId': video_id,
                'Beğeni Sayısı': item['snippet']['topLevelComment']['snippet']['likeCount'],
                'Yanıt Sayısı': item['snippet']['totalReplyCount'],
                'Tarih': item['snippet']['topLevelComment']['snippet']['publishedAt'],
                'Yorum': item['snippet']['topLevelComment']['snippet']['textDisplay']
            }
            comments_data.append(comment_info)

        return comments_data

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        return []

# Yorumları sınıflandırmak için fonksiyon
def siniflandir(comment):
    cleaned_comment = clean_text(comment)
    word_vectors = [word2vec_model.wv[word] for word in cleaned_comment if word in word2vec_model.wv]
    if word_vectors:
        averaged_vector = np.mean(word_vectors, axis=0)
    else:
        averaged_vector = np.zeros(100)
    prediction = model.predict(np.array([averaged_vector]))[0]
    label_index = prediction.argmax()
    return ["Negatif", "Nötr", "Pozitif"][label_index]

def calculate_accuracy(classified_data):
    total_comments = len(classified_data)
    if total_comments == 0:
        return {"Pozitif": 0, "Negatif": 0, "Nötr": 0, "Genel Doğruluk": 0}

    positive_count = sum(1 for comment in classified_data if comment["Sınıflandırma"] == "Pozitif")
    negative_count = sum(1 for comment in classified_data if comment["Sınıflandırma"] == "Negatif")
    neutral_count = sum(1 for comment in classified_data if comment["Sınıflandırma"] == "Nötr")
    general_accuracy = (positive_count + negative_count + neutral_count) / total_comments * 100

    return {
        "Pozitif": (positive_count / total_comments) * 100,
        "Negatif": (negative_count / total_comments) * 100,
        "Nötr": (neutral_count / total_comments) * 100,
        "Genel Doğruluk": general_accuracy
    }