import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Veri temizleme fonksiyonu
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("turkish"))
    stemmer = PorterStemmer()
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return cleaned_tokens