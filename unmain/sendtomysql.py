import pymysql

def server_address(host):
    if host == "localhost":
        user = "root"
        password = "robot123"
        database = "report_ai_cctv"
        port = 3306
    elif host == "10.5.0.8":
        user = "root"
        password = ""
        database = "report_visual_ai"
        port = 3306
    return user, password, database, port


def send_to_server(host):
    try:
        user, password, database, port = server_address(host)
        connection = pymysql.connect(
            host=host, user=user, password=password, database=database, port=port
        )
        cursor = connection.cursor()
        table = "empbro"

        query = f"""
        INSERT INTO {table} (cam, timestamp_start, timestamp_done, elapsed_time, percentage, image_done)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, ("0", 1, 2, 3, 4, None))
        connection.commit()
        print(f"Data berhasil dikirim ")
    except pymysql.MySQLError as e:
        print(f"Error saat mengirim data : {e}")
    finally:
        if "cursor" in locals():
            cursor.close()
        if "connection" in locals():
            connection.close()


if __name__ == "__main__":
    send_to_server("10.5.0.8")
