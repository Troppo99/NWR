import pandas as pd
import numpy as np

# Data awal
data = {"Nama": ["FEBRASARI ALMANIA", "RANI MELIYANA PUTRI", "FARAH NADA MUFIDAH", "DHEO RONALDO SIRAIT", "FEBRY SANDHIKA PUTRA HARAHAP", "DYOTA CANTACYACITTA VIDYADHARI", "CHEKA CAKRECJWARA AL KINDI", "NANA WARTANA"], "Score SKD": [437, 453, 436, 418, 430, 429, 428, 417], "Skor SKB": [390, 325, 320, 315, 295, 295, 295, 305], "Skor Wawancara": [0, 0, 0, 0, 0, 0, 0, 0], "Nilai Akhir": [55.182, 52.445, 50.909, 49.300, 48.973, 48.900, 48.827, 48.627]}

# Membuat DataFrame
df = pd.DataFrame(data)


# Definisikan fungsi untuk menghitung Nilai Akhir
def hitung_nilai_akhir(row):
    skd = row["Score SKD"]
    skb = row["Skor SKB"]
    wawancara = row["Skor Wawancara"]

    nilai_skd = (skd / 550) * 40
    nilai_skb1 = (skb / 500) * 50
    nilai_skb2 = wawancara * 0.3
    nilai_skb_total = (nilai_skb1 + nilai_skb2) * 0.6
    nilai_akhir = nilai_skd + nilai_skb_total
    return nilai_akhir


# Menghitung Nilai Akhir dengan berbagai Skor Wawancara untuk Nana Wartana
skor_wawancara_nana = range(0, 31)  # Dari 0 hingga 30
nilai_akhir_nana = []

for W in skor_wawancara_nana:
    skd = 417
    skb = 305
    nilai_skd = (skd / 550) * 40
    nilai_skb1 = (skb / 500) * 50
    nilai_skb2 = W * 0.3
    nilai_skb_total = (nilai_skb1 + nilai_skb2) * 0.6
    nilai_akhir = nilai_skd + nilai_skb_total
    nilai_akhir_nana.append((W, round(nilai_akhir, 3)))

# Membuat DataFrame hasil simulasi
simulasi_df = pd.DataFrame(nilai_akhir_nana, columns=["Skor Wawancara", "Nilai Akhir"])

# Menentukan Skor Wawancara minimal untuk Nilai Akhir >50.909
target = 50.909
minimal_w = simulasi_df[simulasi_df["Nilai Akhir"] > target]["Skor Wawancara"].min()

print(f"Skor Wawancara minimal untuk Nilai Akhir > {target} adalah: {minimal_w}")

# Menampilkan beberapa baris simulasi
print("\nSimulasi Nilai Akhir berdasarkan Skor Wawancara:")
print(simulasi_df.head(20))
