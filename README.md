# 🧘 YogaAI: AI-Powered Yoga Pose Estimation & Real-Time Posture Coach

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-teal.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Dashboard-black.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**YogaAI**, yapay zeka ve bilgisayarlı görü (Computer Vision) teknolojilerini kullanarak yoga yapanların duruşlarını gerçek zamanlı analiz eden, poz doğruluğunu ölçen ve biyomekanik eklem açılarına göre anlık düzeltme geri bildirimleri sunan modern bir postür rehberidir.

---

## ✨ Özellikler

- **🌐 Modern Web Dashboard**: Tarayıcı üzerinden çalışan, şık karanlık temalı (Dark Mode), canlı kamera akışlı ve görsel grafikli kullanıcı arayüzü.
- **⚡ Gerçek Zamanlı İskelet Takibi**: MediaPipe Pose ile 33 anatomik anahtar nokta tespiti.
- **🎯 Yüksek Doğruluklu Poz Sınıflandırma**: Derin öğrenme (MLP) modeli ile 8 yoga pozu için **%97+ doğruluk oranı**.
- **📐 Akıllı Biyomekanik Geri Bildirim**: 12 kritik eklem açısını altın standart referans açılarıyla kıyaslar ve milimetrik düzeltme önerileri sunar (*"Sol dirseğinizi 15° açın"* vb.).
- **📊 İnteraktif Grafik Analizi**: Chart.js ile mevcut duruş açıları ile ideal poz açılarının anlık karşılaştırma grafiği.
- **🖥️ Masaüstü ve Web Desteği**: Hem tarayıcı tabanlı web paneli (`web_app.py`) hem de OpenCV masaüstü penceresi (`detection2.py`) ile esnek kullanım.

---

## 🧘 Desteklenen Yoga Pozları

1. **Ardha Chandrasana** (Yarım Ay Pozu)
2. **Baddha Konasana** (Kelebek Pozu)
3. **Downward-Facing Dog** (Aşağı Bakan Köpek Pozu)
4. **Natarajasana** (Dansçı Pozu)
5. **Triangle Pose** (Üçgen Pozu)
6. **Utkata Konasana** (Tanrıça Pozu)
7. **Veerabhadrasana** (Savaşçı Pozu)
8. **Vrukshasana** (Ağaç Pozu)

---

## 🛠️ Kurulum

### 1. Depoyu Klonlayın
```bash
git clone <YENI_REPO_URLNIZ>
cd yoga-pose-estimation
```

### 2. Sanal Ortam Oluşturun ve Aktif Edin
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

---

## 🚀 Kullanım

### Seçenek 1: Modern Web Arayüzünü Başlatma (Önerilen)
Tarayıcı tabanlı modern stüdyo arayüzü:
```bash
python web_app.py
```
Ardından tarayıcınızdan **`http://127.0.0.1:5001`** adresine gidin. Bu arayüz üzerinden:
- Canlı web kamerasını başlatabilir,
- Bilgisayarınızdan fotoğraf yükleyebilir,
- Hazır 8 örnek yoga pozunu tek tıkla test edebilirsiniz.

---

### Seçenek 2: Masaüstü OpenCV Modu ile Canlı Kamera
Klasik masaüstü penceresi ile çalıştırma:
```bash
python detection2.py
```

Fotoğraf dosyası üzerinde masaüstü testi için:
```bash
python detection2.py --source dataset2/Triangle/Images/triangle1.jpg
```
*(Çıkmak için ekrandayken `q` tuşuna basabilirsiniz.)*

---

### Seçenek 3: Otomasyon ve Doğrulama Testi
Tüm poz modellerini ve açı hesaplamalarını topluca doğrulamak için:
```bash
python test_pipeline.py
```

---

## 📁 Proje Mimarisi

```
yoga-pose-estimation/
├── app.py                      # Streamlit arayüz alternatifi
├── web_app.py                  # Flask tabanlı modern web arayüz sunucusu
├── detection2.py               # OpenCV masaüstü canlı kamera uygulaması
├── test_pipeline.py            # Otomatik model ve iskelet doğrulama testi
├── dataset.py                  # Veri işleme ve açı çıkarım scripti
├── calculate_ideal.py          # İdeal açı matrisi hesaplayıcı
├── ideal_angles.csv            # 8 pozun ideal eklem açıları tablosu
├── yoga_pose_model.keras       # Eğitilmiş Keras derin öğrenme sınıflandırma modeli
├── yoga_pose_scaler.joblib     # Feature scaler (StandardScaler)
├── yoga_pose_label_encoder.joblib # Sınıf etiket kodlayıcı
├── ideal_images/               # Referans ideal poz görselleri
├── templates/
│   └── index.html              # Modern Dark Mode web paneli (Tailwind + Chart.js)
├── dataset2/                   # Yoga pozları veri seti
└── requirements.txt            # Python bağımlılıkları
```

---

## 📄 Lisans

Bu proje **MIT Lisansı** ile lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakabilirsiniz.
