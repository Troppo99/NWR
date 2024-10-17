import pymysql
import cv2
import numpy as np


def server_address(host):
    if host == "localhost":
        user = "root"
        password = "robot123"
        database = "report_ai_cctv"
        port = 3306
    elif host == "10.5.0.2":
        user = "robot"
        password = "robot123"
        database = "report_ai_cctv"
        port = 3307
    return user, password, database, port


def get_image_from_database(host, table, record_id):
    # Koneksi ke database MySQL
    user, password, database, port = server_address(host)
    try:
        connection = pymysql.connect(
            host=host, user=user, password=password, database=database, port=port
        )
        cursor = connection.cursor()

        # Query untuk mengambil data BLOB
        query = f"SELECT image_done FROM {table} WHERE id = %s"
        cursor.execute(query, (record_id,))
        result = cursor.fetchone()

        if result and result[0]:
            # Ambil data biner dari database
            binary_image = result[0]

            # Mengubah data biner menjadi numpy array
            image_array = np.frombuffer(binary_image, dtype=np.uint8)

            # Decode data biner menjadi gambar
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is not None:
                # Tampilkan gambar menggunakan OpenCV
                cv2.imshow("Retrieved Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Error: Tidak dapat mendecode gambar.")
        else:
            print("No image found for the given record ID.")

    except pymysql.MySQLError as e:
        print(f"Error saat mengakses database: {e}")
    finally:
        if "cursor" in locals():
            cursor.close()
        if "connection" in locals():
            connection.close()


# Panggil fungsi untuk mengambil gambar dari database
id = input("Masukkan ID: ")
get_image_from_database("10.5.0.2", "empbro", id)
