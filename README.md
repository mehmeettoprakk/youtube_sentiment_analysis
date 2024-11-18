# YSA ve GUI Tabanlı Sentiment Analizi

Bu proje, YouTube yorumlarını analiz etmek ve sınıflandırmak için yapay sinir ağları (YSA) ve kullanıcı dostu bir GUI (Grafik Kullanıcı Arayüzü) içerir.

## Projenin Çalıştırılması

### 1. Model Eğitimi

İlk olarak, **`ysa_model.py`** dosyasını çalıştırın. Bu işlem, modeli eğitir ve aşağıdaki dosyaları proje klasöründe oluşturur:

- **`best_model_structure.png`**: Eğitilen modelin yapısını görselleştirir.
- **`best_model_summary.txt`**: Modelin özetini metin formatında sağlar.
- **`labelencoder.pkl`**: Etiketleme için kullanılan LabelEncoder objesini içerir.
- **`model_comparison.png`**: Farklı modellerin karşılaştırma grafiğini gösterir.
- **`sentiment_model.h5`**: Eğitim sonrası oluşturulan en iyi modeli içerir.
- **`word2vec_model.model`**: Word2Vec tabanlı vektörleme modelini içerir.

Eğitim sırasında modelin performansı, karşılaştırma grafiği ve yapı özeti otomatik olarak kaydedilir.

### 2. Kullanıcı Arayüzü

Model eğitimi tamamlandıktan sonra, **`gui.py`** dosyasını çalıştırarak kullanıcı arayüzünü başlatın. Bu arayüz, yorumların analiz edilmesi ve sınıflandırılması için kullanılır.
