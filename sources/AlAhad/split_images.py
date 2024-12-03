import os
import shutil
import math


def move_images(source_dir, dest_dir, percentage=30):
    if not os.path.exists(source_dir):
        print(f"Direktori sumber tidak ditemukan: {source_dir}")
        return
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Direktori tujuan dibuat: {dest_dir}")

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
    all_files = sorted([f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and os.path.splitext(f)[1].lower() in image_extensions])
    total_files = len(all_files)
    if total_files == 0:
        print("Tidak ada file gambar yang ditemukan di direktori sumber.")
        return

    num_to_move = math.ceil((percentage / 100) * total_files)
    step = total_files / num_to_move
    indices_to_move = [int(math.floor(step * i)) for i in range(num_to_move)]
    indices_to_move = [idx for idx in indices_to_move if idx < total_files]
    for idx in indices_to_move:
        file_name = all_files[idx]
        src_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)

        if os.path.exists(dest_path):
            print(f"File sudah ada di direktori tujuan, dilewati: {file_name}")
            continue

        shutil.move(src_path, dest_path)
        print(f"File dipindahkan: {file_name}")

    print(f"Pindahkan selesai: {len(indices_to_move)} dari {total_files} file.")


if __name__ == "__main__":
    base_dir = r"D:\NWR\sources\AlAhad\images\wallpaper3"
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")
    move_images(train_dir, valid_dir, percentage=30)
