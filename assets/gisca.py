import requests
import json


url = "https://10.8.0.108:7186/api/gina/notif-create"
headers = {"GCC-API-KEY": "332100185", "Content-Type": "application/json"}
payload = {"whatsapp_number": "082133732051", "message": ("PENGAJUAN RESIGN\n" "Hi, RIZKI.\n\n" "Berikut informasi detail pengajuan resign\n\n" "Tanggal pengajuan\t:\tJumat, 18 Oktober 2024"), "attachment_path": "", "app": "visual-ai-5s", "modul": "line-detection"}

try:
    response = requests.post(url, headers=headers, data=json.dumps(payload), verify=False, timeout=10)
    if response.status_code == 200:
        print("Pesan berhasil dikirim.")
        print("Respons:", response.json())
    else:
        print(f"Terjadi kesalahan. Status Code: {response.status_code}")
        print("Respons:", response.text)
except requests.exceptions.RequestException as e:
    print(f"Error saat mengirim permintaan: {e}")
