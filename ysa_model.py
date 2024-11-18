import os
import numpy as np
import pandas as pd
import pickle  
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tokenizer import clean_text
import matplotlib.pyplot as plt

# Model dosya yolları
WORD2VEC_PATH = "word2vec_model.model"
MLP_MODEL_PATH = "sentiment_model.h5"
LABEL_ENCODER_PATH = "labelencoder.pkl"

def save_label_encoder(label_encoder, path):
    """
    LabelEncoder'ı bir dosyaya kaydetmek için fonksiyon.
    """
    with open(path, "wb") as file:
        pickle.dump(label_encoder, file)
    print(f"LabelEncoder kaydedildi: {path}")

def load_label_encoder(path):
    """
    Kaydedilmiş LabelEncoder'ı yüklemek için fonksiyon.
    """
    with open(path, "rb") as file:
        label_encoder = pickle.load(file)
    print(f"LabelEncoder yüklendi: {path}")
    return label_encoder

def train_multiple_models(X_train, X_test, y_train, y_test, input_shape):
    models = []
    histories = []
    test_accuracies = []

    for i, layers in enumerate([
        [128, 64, 64],  # Model 1
        [256, 128, 64], # Model 2
        [64, 32]        # Model 3
    ], start=1):
        try:
            # Model oluşturma
            model = Sequential()
            model.add(Dense(layers[0], activation='relu', input_shape=input_shape))
            model.add(Dropout(0.3))
            for layer in layers[1:]:
                model.add(Dense(layer, activation='relu'))
                model.add(Dropout(0.3))
            model.add(Dense(3, activation='softmax'))

            
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            
            print(f"Model {i} eğitiliyor...")
            history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)

            # Model test
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Model {i} Test Accuracy: {test_accuracy:.4f}")

            # Modelleri ve geçmişleri listeye ekleme
            models.append(model)
            histories.append(history)
            test_accuracies.append(test_accuracy)
        except Exception as e:
            print(f"Model {i} eğitilirken hata oluştu: {e}")

    # Hiçbir model başarıyla eğitilemezse hata fırlatma 
    if not histories:
        raise ValueError("Hiçbir model başarıyla eğitilemedi.")

    # En iyi modeli seçme
    best_model_index = np.argmax(test_accuracies)
    best_model = models[best_model_index]

    print(f"En iyi model: Model {best_model_index + 1} Test Accuracy: {test_accuracies[best_model_index]:.4f}")

    # En iyi modelin özetini ve yapısını kaydetme
    with open(f"best_model_summary.txt", "w", encoding="utf-8") as f:
        best_model.summary(print_fn=lambda x: f.write(x + "\n"))
    plot_model(best_model, to_file="best_model_structure.png", show_shapes=True)
    print("En iyi modelin yapısı ve özeti kaydedildi.")

    # Karşılaştırmalı grafiği çizme ve kaydetme
    try:
        plot_comparison_graph(histories)
    except Exception as e:
        print(f"Grafik oluşturulurken hata oluştu: {e}")

    return best_model, models, histories

def plot_comparison_graph(histories):
    """
    3 modelin doğrulama doğruluk oranlarını karşılaştırmalı olarak çizen ve kaydeden fonksiyon.
    """
    try:
        plt.figure(figsize=(10, 6))
        for i, history in enumerate(histories, start=1):
            plt.plot(history.history['val_accuracy'], label=f'Model {i} Validation Accuracy')

        plt.title("Model Validation Accuracy Comparison", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Validation Accuracy", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        # Grafiği kaydetme
        output_path = "model_comparison.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Model karşılaştırma grafiği kaydedildi: {output_path}")
    except Exception as e:
        print(f"Grafik çizilirken hata oluştu: {e}")

def train_or_load_models():
    # Eğer modeller varsa yükleme
    if os.path.exists(WORD2VEC_PATH) and os.path.exists(MLP_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        print("Eğitilmiş modeller ve LabelEncoder yükleniyor...")
        word2vec_model = Word2Vec.load(WORD2VEC_PATH)
        model = load_model(MLP_MODEL_PATH)
        label_encoder = load_label_encoder(LABEL_ENCODER_PATH)
        return model, word2vec_model, label_encoder

    # Eğer modeller yoksa eğitme
    print("Modeller bulunamadı, yeni modeller eğitiliyor...")
    try:
        # Veri setini yükleme
        data = pd.read_csv("comment_dataset.csv", encoding='utf-16')
    except UnicodeDecodeError:
        data = pd.read_csv("comment_dataset.csv", encoding='utf-8')
    except Exception as e:
        raise ValueError(f"Dosya yüklenirken hata oluştu: {e}")

    # Veri temizleme ve dönüştürme
    data["Durum"] = data["Durum"].str.strip()
    data["Durum"] = data["Durum"].replace({"Olumlu": "Pozitif", "Olumsuz": "Negatif", "Nötr": "Nötr"})
    data = data.dropna(subset=["Görüş", "Durum"])

    # Veri dengesizliğini giderme
    positive_data = data[data["Durum"] == "Pozitif"]
    neutral_data = data[data["Durum"] == "Nötr"]
    negative_data = data[data["Durum"] == "Negatif"]

    neutral_data_downsampled = resample(neutral_data, replace=True, n_samples=len(positive_data), random_state=42)
    negative_data_upsampled = resample(negative_data, replace=True, n_samples=len(positive_data), random_state=42)

    balanced_data = pd.concat([positive_data, neutral_data_downsampled, negative_data_upsampled])
    balanced_data["cleaned_comment"] = balanced_data["Görüş"].apply(clean_text)

    # Word2Vec modeli eğitme
    word2vec_model = Word2Vec(sentences=balanced_data["cleaned_comment"], vector_size=100, window=5, min_count=1, workers=4)

    # Yorumları vektörlere dönüştürme
    X = []
    for tokens in balanced_data["cleaned_comment"]:
        word_vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
        if word_vectors:
            X.append(np.mean(word_vectors, axis=0))
        else:
            X.append(np.zeros(100))
    X = np.array(X)

    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(balanced_data["Durum"])
    y = to_categorical(y, num_classes=3)

    
    save_label_encoder(label_encoder, LABEL_ENCODER_PATH)

    # Eğitim ve test veri setlerini ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3 farklı modeli eğit ve en iyisini seç
    best_model, all_models, training_histories = train_multiple_models(X_train, X_test, y_train, y_test, (100,))

    # En iyi modeli ve Word2Vec modelini kaydetme
    word2vec_model.save(WORD2VEC_PATH)
    best_model.save(MLP_MODEL_PATH)
    print(f"En iyi model kaydedildi: {MLP_MODEL_PATH}")
    print(f"Word2Vec modeli kaydedildi: {WORD2VEC_PATH}")

    return best_model, word2vec_model, label_encoder

# Model eğitme veya yükleme
model, word2vec_model, label_encoder = train_or_load_models()
