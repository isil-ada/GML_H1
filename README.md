# 📧 Grafik Tabanlı E-posta Sınıflandırma (Graph Machine Learning)

Bu proje, e-postaları **kişisel (personal)** veya **profesyonel (professional)** olarak sınıflandırmak için **graf tabanlı makine öğrenmesi yöntemlerini** kullanmaktadır.

Projede e-postalar bir **graf yapısı** içerisinde temsil edilir ve farklı **graf embedding yöntemleri** ile düğüm temsilleri öğrenilir. Daha sonra bu temsiller kullanılarak sınıflandırma yapılır ve farklı yöntemlerin performansı karşılaştırılır.

Proje dört ana aşamadan oluşmaktadır:

1. Veri Toplama ve Ön İşleme
2. Graf Oluşturma
3. Embedding (Temsil Öğrenme)
4. Sınıflandırma ve Performans Değerlendirmesi

---

# 🚀 Proje Aşamaları

## 1️⃣ PHASE 1: Veri Yükleme ve Ön İşleme

Bu aşamada ham e-posta verisi yüklenir ve analiz için hazırlanır.

### Yapılan İşlemler

* `emails.csv` veri seti yüklenir.
* Dosya yollarından **klasör bilgisi (folder)** çıkarılır.
* E-postaların içeriği temizlenir.
* Gereksiz başlıklar kaldırılır:

Örnek kaldırılan başlıklar:

* Message-ID
* Date
* From
* To
* Subject
* Content-Type

Ayrıca:

* İmzalar temizlenir
* Gereksiz boşluklar kaldırılır
* Çok kısa e-postalar filtrelenir

---

### 🏷 Veri Etiketleme

E-postalar klasörlerine göre etiketlenir.

| Etiket | Anlam               |
| ------ | ------------------- |
| 0      | Profesyonel e-posta |
| 1      | Kişisel e-posta     |

Kullanılan klasörler:

**Profesyonel klasörler**

* sent
* sent_items
* _sent_mail
* all_documents
* discussion_threads

**Kişisel klasörler**

* personal
* inbox
* notes_inbox

---

### 📊 Örnekleme (Sampling)

Dengeli bir veri seti oluşturmak için:

| Sınıf       | Örnek Sayısı |
| ----------- | ------------ |
| Profesyonel | 2500         |
| Kişisel     | 2500         |

Toplam veri:

**5000 e-posta**

---

# 🌐 2️⃣ PHASE 2: Graf Oluşturma

Her e-posta graf içerisinde **bir düğüm (node)** olarak temsil edilir.

E-postalar arasındaki ilişkiler **kenar (edge)** olarak eklenir.

İki farklı yöntem kullanılmıştır.

---

## 🔹 Named Entity Recognition (NER)

E-postalar içerisindeki varlıklar **spaCy** kullanılarak çıkarılır.

Kullanılan entity tipleri:

* PERSON
* ORG
* GPE

Eğer iki e-posta **en az iki ortak entity içeriyorsa**, aralarında bir kenar oluşturulur.

Edge tipi:

```
type = "ner"
```

Sonuç:

* **NER edge sayısı:** 25,156

---

## 🔹 Bag-of-Words Benzerliği

E-postalar **TF-IDF** ile vektörleştirilir.

Ardından **cosine similarity** hesaplanır.

Eğer iki e-posta arasındaki benzerlik:

```
cosine similarity > 0.7
```

ise aralarında kenar oluşturulur.

Edge tipi:

```
type = "bow"
```

Sonuç:

* **BoW edge sayısı:** 36,422

---

## 📊 Graf Özeti

| Özellik         | Değer  |
| --------------- | ------ |
| Node sayısı     | 5000   |
| Edge sayısı     | 56,037 |
| NER edges       | 25,156 |
| BoW edges       | 36,422 |
| Ortalama derece | 22.41  |

---

# 🧠 3️⃣ PHASE 3: Graph Embedding Yöntemleri

Bu aşamada graf üzerindeki düğümler için **vektör temsilleri (embedding)** öğrenilir.

Her yöntem **64 boyutlu embedding** üretir.

---

## 🔹 DeepWalk

DeepWalk yöntemi:

1. Graf üzerinde **random walk** üretir
2. Walk’ları **Word2Vec Skip-Gram** modeli ile eğitir

Çıktı:

```
DeepWalk embedding boyutu: (5000, 64)
```

---

## 🔹 Node2Vec

Node2Vec, DeepWalk’un geliştirilmiş versiyonudur.

Farkı:

* **Biaslı random walk** kullanması

Parametreler:

```
dimensions = 64
walk_length = 10
num_walks = 10
p = 1
q = 0.5
```

Çıktı:

```
Node2Vec embedding boyutu: (5000, 64)
```

---

## 🔹 SiGraC (Signed Graph Embedding)

Bu yöntemde graf:

* **pozitif kenarlar**
* **negatif kenarlar**

olarak ayrılır.

Bu projede:

| Edge tipi | Anlam   |
| --------- | ------- |
| NER       | Pozitif |
| BoW       | Negatif |

Ardından:

1. Adjacency matrix oluşturulur
2. Normalize edilir
3. Signed embedding hesaplanır
4. PCA ile 64 boyuta indirgenir

Çıktı:

```
SiGraC embedding boyutu: (5000, 64)
```

---

## 🔹 Graph Neural Network (GCN)

Bu projede **2 katmanlı Graph Convolutional Network** kullanılmıştır.

Model mimarisi:

```
Input Features
      ↓
GCNConv Layer
      ↓
ReLU
      ↓
Dropout
      ↓
GCNConv Layer
      ↓
Output
```

Eğitim parametreleri:

```
Epoch: 100
Learning Rate: 0.01
Hidden Layer: 64
```

Çıktı:

```
GCN embedding boyutu: (5000, 64)
```

---

# 📊 4️⃣ PHASE 4: Sınıflandırma ve Değerlendirme

Embedding'ler kullanılarak **Logistic Regression** modeli eğitilir.

Veri bölünmesi:

| Veri  | Oran |
| ----- | ---- |
| Train | %80  |
| Test  | %20  |

Kullanılan metrikler:

* Accuracy
* Precision
* Recall
* F1 Score

---

# 📈 Sonuçlar

| Yöntem   | Accuracy  | F1 Score  |
| -------- | --------- | --------- |
| DeepWalk | 0.631     | 0.620     |
| Node2Vec | 0.628     | 0.615     |
| SiGraC   | 0.615     | 0.596     |
| GCN      | **0.655** | **0.653** |

### 📌 Yorum

En yüksek performansı **Graph Neural Network (GCN)** elde etmiştir.

Bu sonuç, **derin graf modellerinin graf yapısını daha iyi öğrenebildiğini** göstermektedir.

---

# 🛠 Kullanılan Teknolojiler

* Python
* Pandas
* NumPy
* spaCy
* NetworkX
* Scikit-learn
* Gensim
* Node2Vec
* PyTorch
* PyTorch Geometric

---

# ⚙️ Kurulum

Gerekli kütüphaneleri yüklemek için:

```bash
pip install spacy scikit-learn networkx gensim node2vec torch torch-geometric
```

spaCy modelini indirmek için:

```bash
python -m spacy download en_core_web_sm
```

---

# ▶️ Notebook Çalıştırma

Notebook'u çalıştırmak için:

```
GML_H1.ipynb
```

dosyasını **Google Colab** veya **Jupyter Notebook** ile açabilirsiniz.

Veri seti aynı klasörde bulunmalıdır:

```
emails.csv
```
