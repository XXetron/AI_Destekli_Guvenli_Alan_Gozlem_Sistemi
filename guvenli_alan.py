import os

# TensorFlow uyarılarını gizle
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import pandas as pd
from deepface import DeepFace
from datetime import datetime


# AYARLAR

video_yolu = 0
db_yolu = "veritabani"
log_dosyasi = "guvenlik_kayitlari.csv"
supheli_klasoru = "supheli_fotograflar"  # Fotoğrafların kaydedileceği yer

# Hareket Hassasiyeti (Daha küçük sayı = Daha hassas)
MIN_ALAN = 500

# Renkler (B, G, R)
RENK_HAREKET = (0, 255, 255)  # Sarı
RENK_TANIMLANAN = (0, 255, 0)  # Yeşil
RENK_BILINMEYEN = (0, 0, 255)  # Kırmızı


# 1. Başlatma ve Dosya Hazırlığı

# Log dosyası yoksa oluştur
if not os.path.exists(log_dosyasi):
    with open(log_dosyasi, "w") as f:
        f.write("Tarih,Saat,Durum,Kisi,Fotograf_Yolu\n")

# Şüpheli fotoğraflar klasörü yoksa oluştur
if not os.path.exists(supheli_klasoru):
    os.makedirs(supheli_klasoru)

video_yakalama = cv2.VideoCapture(video_yolu)
arka_plan_cikartici = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

if not video_yakalama.isOpened():
    print("HATA: Kamera açılamadı.")
    exit()

kare_sayisi = 0
yuz_tanima_araligi = 30  # Her 30 karede bir (1 saniye) tara
son_taninan_isim = ""

print("Sistem devrede... 'q' tuşuna basarak çıkabilirsiniz.")


# 2. Ana Döngü

while True:
    ret, kare = video_yakalama.read()
    if not ret:
        break

    kare_sayisi += 1
    kare = cv2.flip(kare, 1)  # Aynalama

    # HAREKET ALGILAMA
    gri_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    gri_kare = cv2.GaussianBlur(gri_kare, (21, 21), 0)
    on_plan_maskesi = arka_plan_cikartici.apply(gri_kare)
    _, esiklenmis_maske = cv2.threshold(on_plan_maskesi, 200, 255, cv2.THRESH_BINARY)
    esiklenmis_maske = cv2.dilate(esiklenmis_maske, None, iterations=2)
    konturlar, _ = cv2.findContours(esiklenmis_maske.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hareket_var = False
    for kontur in konturlar:
        if cv2.contourArea(kontur) < MIN_ALAN:
            continue
        hareket_var = True
        (x, y, w, h) = cv2.boundingRect(kontur)
        cv2.rectangle(kare, (x, y), (x + w, y + h), RENK_HAREKET, 2)

    # YÜZ TANIMA KAYIT VE FOTOĞRAF ÇEKME
    if hareket_var and (kare_sayisi % yuz_tanima_araligi == 0):
        print("Yüz taranıyor...")

        try:
            sonuclar = DeepFace.find(img_path=kare,
                                     db_path=db_yolu,
                                     model_name="VGG-Face",
                                     enforce_detection=False,
                                     silent=True)

            anlik_zaman = datetime.now()
            tarih_str = anlik_zaman.strftime("%Y-%m-%d")
            saat_str = anlik_zaman.strftime("%H:%M:%S")
            dosya_saat = anlik_zaman.strftime("%H-%M-%S")  # Dosya adında ':' kullanılamaz

            if len(sonuclar) > 0:
                df = sonuclar[0]

                #  SENARYO 1: TANINAN KİŞİ
                if not df.empty:
                    tam_yol = df.iloc[0]['identity']
                    isim = os.path.basename(os.path.dirname(tam_yol))
                    son_taninan_isim = f"Tanindi: {isim}"

                    # Loga Yaz
                    with open(log_dosyasi, "a") as f:
                        f.write(f"{tarih_str},{saat_str},YETKILI,{isim},-\n")
                    print(f"Yetkili Girişi: {isim}")

                # SENARYO 2: BİLİNMEYEN KİŞİ FOTOĞRAF ÇEK
                else:
                    son_taninan_isim = "Bilinmeyen Kisi"

                    # Fotoğrafı Kaydet
                    foto_adi = f"supheli_{tarih_str}_{dosya_saat}.jpg"
                    foto_yolu = os.path.join(supheli_klasoru, foto_adi)
                    cv2.imwrite(foto_yolu, kare)  # O anki kareyi dosyaya yaz

                    # Loga Yaz Fotoğraf yolunu da ekle
                    with open(log_dosyasi, "a") as f:
                        f.write(f"{tarih_str},{saat_str},SUPHELI,Bilinmeyen,{foto_adi}\n")

                    print(f"UYARI: Bilinmeyen kişi! Fotoğraf kaydedildi: {foto_adi}")

        except Exception as e:
            print(f"Hata: {e}")

    # EKRAN GÖSTERİMİ
    durum_yazisi = "Durum: Hareketli" if hareket_var else "Durum: Sakin"
    cv2.putText(kare, durum_yazisi, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RENK_HAREKET, 2)

    if son_taninan_isim:
        renk = RENK_TANIMLANAN if "Tanindi" in son_taninan_isim else RENK_BILINMEYEN
        cv2.putText(kare, son_taninan_isim, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, renk, 2)

    cv2.imshow("Guvenli Alan Gozlem", kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_yakalama.release()
cv2.destroyAllWindows()