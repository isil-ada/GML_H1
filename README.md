# Graf Tabanlı E-posta Sınıflandırması
**Enron Veri Seti — GML Ödev 1**

---

## Genel Bakış

Bu proje, Enron E-posta Veri Seti üzerinde graf tabanlı bir sınıflandırma hattı uygulamaktadır. Her e-posta bir graf düğümü olarak temsil edilir; kenarlar ortak Adlandırılmış Varlıklar (NER) ve içerik benzerliği (BoW/TF-IDF) kriterlerine göre oluşturulur. Dört gömme yöntemi, Lojistik Regresyon sınıflandırıcı ile karşılaştırılır.

---

## Proje Yapısı

Boru hattı dört aşamadan oluşur:

- Aşama 1: Veri Edinimi ve Ön İşleme
- Aşama 2: Graf Oluşturma (NER + BoW kenarları)
- Aşama 3: Gömme Teknikleri (DeepWalk, Node2Vec, SiGraC, GCN)
- Aşama 4: Sınıflandırma ve Değerlendirme

---

## Aşama 1 — Veri Edinimi ve Ön İşleme

### Veri Seti

- Kaynak: Enron E-posta Veri Seti (`emails.csv`)
- Ham boyut: 843.222 satır; `file` ve `message` sütunlarından oluşur.

### Klasör Tabanlı Etiketleme

E-postalar iki kategoriye ayrılır:

- **Profesyonel (etiket = 0):** `sent`, `sent_items`, `_sent_mail`, `all_documents`, `discussion_threads`
- **Kişisel (etiket = 1):** `personal`, `inbox`, `notes_inbox`

### Temizleme Adımları

- E-posta başlıkları kaldırıldı (`From`, `To`, `Subject`, `X-*` alanları, MIME başlıkları)
- Ayırıcı satırlar ve imzalar silindi
- Boşluklar normalize edildi
- Gövde uzunluğu 20 karakterin altındaki e-postalar elendi

Nihai çalışma örneklemi: **5.000 e-posta** (sınıf başına 2.500, katmanlı örnekleme).

---

## Aşama 2 — Graf Oluşturma

### NER Kenarları

spaCy (`en_core_web_sm`) modeli her e-postanın gövdesinden (ilk 500 karakter) `PERSON`, `ORG` ve `GPE` türünde adlandırılmış varlıkları çıkarır. İki e-posta en az 2 ortak benzersiz varlık paylaşıyorsa aralarında bir kenar oluşturulur.

### BoW/TF-IDF Kenarları

Her e-posta için TF-IDF vektörü hesaplanır (en fazla 5.000 özellik, durdurma sözcükleri çıkarılmış). Kosinüs benzerliği 0,70 eşiğini aşıyorsa iki e-posta arasında kenar eklenir. Bellek yönetimi için benzerlik 500'lük partiler hâlinde hesaplanır.

### Birleşik Graf

NER ve BoW kenarları birleştirilir (birleşim). Elde edilen `G = (V, E)` grafı **5.000 düğüm** içerir; her düğüm bir e-postaya karşılık gelir.

---

## Aşama 3 — Gömme Teknikleri

### Düğüm Özellik Matrisi

TF-IDF matrisi `TruncatedSVD` (LSA) ile 64 boyuta indirgenir. Bu `X` matrisi SiGraC ve GCN için giriş olarak kullanılır.

---

### DeepWalk

Özel rastgele yürüyüş implementasyonu (düğüm başına 10 yürüyüş, uzunluk 10). Gensim Word2Vec Skip-gram modeli yürüyüş dizileri üzerinde eğitilir.

| Parametre | Değer |
|---|---|
| Gömme boyutu | 64 |
| Pencere boyutu | 5 |
| Epoch sayısı | 5 |

---

### Node2Vec

`node2vec` kütüphanesi. BFS (yerel) ve DFS (küresel) keşfini dengeleyen yanlı rastgele yürüyüşler kullanır.

| Parametre | Değer |
|---|---|
| Gömme boyutu | 64 |
| Yürüyüş uzunluğu | 10 |
| Düğüm başına yürüyüş | 10 |
| p (geri dönüş) | 1 |
| q (iç-dış, DFS ağırlıklı) | 0,5 |

---

### SiGraC — Çoklu Yakınlık Ölçüleri

SiGraC (Coşkun ve Koyutürk, *Bioinformatics* 2021), yakınlık ağırlıklı komşuluk matrisleri hesaplar ve düğüm özelliklerini iki katmanlı graf konvolüsyonu ile yayar. GCN'den farklı olarak öğrenilecek ağırlık matrisi yoktur; yakınlık matrisi yapı ağırlıklarını doğrudan taşır.

Uygulama — kapalı-form yayılım (eğitim döngüsü yoktur):

```
A_hat = D^{-1/2} (A_prox + I) D^{-1/2}
H1    = ReLU(A_hat @ X)
H2    = ReLU(A_hat @ H1)   ← nihai gömme, L2 normalize
```

Beş farklı yakınlık ölçüsü değerlendirilir; her ölçü ayrı bir gömme vektörü üretir:

| Ölçü | Formül | Davranış |
|---|---|---|
| **CN** — Ortak Komşular | `\|N(u) ∩ N(v)\|` | Ham ortak komşu sayısı. Yoğun bağlı çiftlere yüksek skor verir. |
| **AA** — Adamic-Adar | `Σ 1/log(derece(w))` | Yüksek dereceli ortak komşuları cezalandırır; nadir köprü düğümleri tercih eder. |
| **RA** — Kaynak Tahsisi | `Σ 1/derece(w)` | AA'ya benzer ancak merkez düğümler için daha güçlü ceza uygular. |
| **HPI** — Merkez Destekli | `\|N(u)∩N(v)\| / min(derece)` | Merkez düğümler için yüksek skor üretir. |
| **HDI** — Merkez Baskılayan | `\|N(u)∩N(v)\| / max(derece)` | Baskın merkez bağlantıları cezalandırır. Varsayılan SiGraC ölçüsü. |

Her ölçü, değerlendirme tablosunda ayrı bir satır olarak yer alır. Ortak komşusu olmayan kenarlara graf bağlantısını korumak amacıyla `1,0` ikili ağırlık atanır.

---

### GNN — Graf Evrişimsel Ağ

PyTorch Geometric ile oluşturulmuş 2 katmanlı GCN. Düğüm özellikleri 64 boyutlu TF-IDF/SVD vektörleridir. Çapraz entropi kaybı ile uçtan uca eğitilir (%80/%20 eğitim/test bölümü). Gömmeler eğitim sonrasında birinci konvolüsyon katmanından alınır.

| Parametre | Değer |
|---|---|
| Gizli kanal | 64 |
| Çıkış | 2 sınıf |
| Optimizer | Adam (lr=0,01, weight_decay=5e-4) |
| Dropout | 0,5 |
| Epoch | 100 |

---

## Aşama 4 — Sınıflandırma ve Değerlendirme

### Sınıflandırıcı

1–3. yöntemlerden elde edilen gömmeler üzerinde Lojistik Regresyon (`scikit-learn`, `max_iter=1000`, `random_state=42`) eğitilir. GCN uçtan uca değerlendirilir. Eğitim/test bölümü: %80/%20, katmanlı.

### Metrikler

- Doğruluk (Accuracy)
- Ağırlıklı F1-Skoru
- Eğitim Süresi (gömme süresi + sınıflandırıcı fit süresi)

### Sonuçlar

| Yöntem | Doğruluk | F1-Skoru | Notlar |
|---|---|---|---|
| DeepWalk | ~0,60 | ~0,58 | Rastgele yürüyüş tabanlı baz |
| Node2Vec | ~0,59 | ~0,57 | DFS ağırlıklı (q=0,5) |
| SiGraC [CN] | — | — | Ortak Komşular |
| SiGraC [AA] | — | — | Adamic-Adar |
| SiGraC [RA] | — | — | Kaynak Tahsisi |
| SiGraC [HPI] | — | — | Merkez Destekli |
| SiGraC [HDI] | — | — | Merkez Baskılayan (varsayılan) |
| GNN (GCN) | ~0,67 | ~0,67 | Genel en iyi sonuç |

> SiGraC sonuç hücreleri notebook çalıştırıldıktan sonra gerçek değerlerle doldurulacaktır.

---

## Kurulum ve Gereksinimler

### Ortam

Google Colab (Python 3.10+). Çalıştırmadan önce `emails.csv` dosyasını `/content/` dizinine yükleyin.

### Bağımlılıklar

```bash
pip install spacy scikit-learn networkx gensim node2vec torch torch-geometric
python -m spacy download en_core_web_sm
```

### Çalıştırma Sırası

Tüm hücreler yukarıdan aşağıya doğru sırasıyla çalıştırılmalıdır. Her aşama bir önceki aşamanın değişkenlerine bağımlıdır; hücre atlanmamalıdır.

---

## Teknik Notlar

### SiGraC İmplementasyon Ayrıntısı

Orijinal makale, öğrenilecek parametre içermeyen kapalı-form spektral konvolüsyon kullanır. Önceki implementasyonda gözetimsiz bir yeniden oluşturma kaybı (`sigmoid(H H^T) ≈ A_hat`) kullanılıyordu; bu yaklaşım seyrek graflarda gradyan çöküşüne neden oldu: `A_hat` kimlik matrisine yaklaştığında gömmeler sıfıra yakınsadı ve sınıflandırıcı yalnızca çoğunluk sınıfını tahmin eder hâle geldi. Mevcut implementasyon, normalize edilmiş yakınlık matrisi üzerinden doğrudan özellik yayılımı kullanmaktadır; bu hem makale tasarımına uygundur hem de bu bozunumu önler.

### Yalıtılmış Kenar Çiftleri için Ağırlık

İki bağlı düğüm hiç ortak komşu paylaşmıyorsa AA, RA, HPI ve HDI ölçüleri 0 skoru döndürerek kenarı ağırlıklı graftan etkin şekilde çıkarır. Bu durumda graf bağlantısını korumak amacıyla `1,0` (ikili komşuluk) yedek ağırlık atanır.

### Bellek Optimizasyonu

Yakınlık matrisi ve normalize edilmiş `A_hat`, işlem boyunca `scipy` seyrek matris formatında tutulur. Yoğun matrise dönüşüm yapılmaz. Özellik yayılımı (`A_hat @ X`), kenar yoğunluğunun düşük olduğu graflarda verimli olan seyrek-yoğun matris çarpımı ile gerçekleştirilir.

---

## Kaynaklar

- Coşkun, M. ve Koyutürk, M. (2021). Node similarity-based graph convolution for link prediction in biological networks. *Bioinformatics*, 37(23), 4501–4508.
- SiGraC kaynak kodu: https://github.com/mustafaCoskunAgu/SiGraC
- Enron E-posta Veri Seti: https://www.cs.cmu.edu/~enron/
