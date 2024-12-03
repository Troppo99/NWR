import os
import yaml

# Daftar kelas utama yang akan digunakan untuk standarisasi
master_names = [
    "ADE EKA ARDIAN",
    "AI SITI NUROHMAH",
    "AMI GINANJAR",
    "ANANDA PUTRI LIANI",
    "DEDE FAUJIAWATUL MUNAWAROH",
    "DESI NATALIA",
    "DETI DENIAWATI",
    "DHEA ASTRI SABRINA",
    "DITA NITI YUNIARTI",
    "ENDE NINA",
    "ERNA NITI HERLIAWATI",
    "FIFI APRILISTIYANTI",
    "IIS",
    "IRWAN GUNAWAN",
    "KRIASIAN NOVITA",
    "LARAS SRI PADILAH",
    "LINA NURYANTI",
    "LUSI LESTARI",
    "NANI NURYANI",
    "NANI SUMIATI",
    "NENG TIA SITI PARIJAH",
    "NOVI GUSMAYYANTI",
    "NURHASANAH",
    "OPI SOFIA",
    "RAHMAN",
    "RISNAWATI",
    "ROS",
    "SELI MARSELA",
    "SEPTIANI ULFIANITA",
    "SISKA FITRIYANI",
    "SITI HOTIJAH",
    "SITI ILMIATI",
    "SITI SUMIYATI",
    "SITI UMI SOLEHA",
    "SRI ANZANI",
    "SURYADI",
    "WIWI CARWITI",
    "YULIANA DEWI",
]


def update_labels(label_dir, index_mapping):
    """
    Memperbarui indeks kelas dalam file label sesuai dengan index_mapping.
    """
    for root, _, files in os.walk(label_dir):
        for file_name in files:
            if file_name.endswith(".txt"):
                label_file_path = os.path.join(root, file_name)
                with open(label_file_path, "r") as f:
                    lines = f.readlines()
                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        # Memetakan class_id lama ke class_id baru
                        new_class_id = index_mapping.get(class_id)
                        if new_class_id is not None:
                            parts[0] = str(new_class_id)
                            updated_line = " ".join(parts)
                            updated_lines.append(updated_line)
                        else:
                            print(
                                f"Warning: class_id {class_id} tidak ditemukan dalam index_mapping. File: {label_file_path}"
                            )
                    else:
                        updated_lines.append(line.strip())
                # Menulis kembali file label dengan indeks yang diperbarui
                with open(label_file_path, "w") as f:
                    f.write("\n".join(updated_lines))


def process_dataset(dataset_path):
    """
    Memproses dataset dalam folder tertentu untuk menyesuaikan data.yaml dan label.
    """
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.isfile(data_yaml_path):
        print(f"data.yaml TIDAK ditemukan di folder {dataset_path}")
        return
    # Membaca data.yaml
    with open(data_yaml_path, "r") as file:
        data = yaml.safe_load(file)
    names = data.get("names", [])
    old_nc = data.get("nc", len(names))
    print(f"Proses folder: {dataset_path}")
    print(f"Jumlah kelas sebelum: {old_nc}, Kelas: {names}")
    # Membuat mapping dari nama kelas lama ke indeks lama
    old_name_to_index = {name: idx for idx, name in enumerate(names)}
    # Membuat mapping dari indeks lama ke indeks baru
    index_mapping = {}
    for name, old_index in old_name_to_index.items():
        if name in master_names:
            new_index = master_names.index(name)
            index_mapping[old_index] = new_index
        else:
            print(f"Nama '{name}' tidak ditemukan dalam master_names. File: {data_yaml_path}")
    # Memperbarui data.yaml dengan master_names dan nc = 38
    data["names"] = master_names
    data["nc"] = len(master_names)
    # Menyimpan data.yaml yang telah diperbarui
    with open(data_yaml_path, "w") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    print(f"data.yaml diperbarui di folder {dataset_path}")
    # Memperbarui label dalam folder train/labels, valid/labels, dan test/labels
    for split in ["train", "valid", "test"]:
        labels_dir = os.path.join(dataset_path, split, "labels")
        if os.path.isdir(labels_dir):
            update_labels(labels_dir, index_mapping)
            print(f"Label diperbarui di {labels_dir}")
        else:
            print(f"Folder labels tidak ditemukan di {labels_dir}")
    print(f"Jumlah kelas setelah: {data['nc']}, Kelas: {data['names']}\n")


# Path ke direktori yang berisi folder dataset
parent_directory = r"D:\SBHNL\Images\BSML\Datasets\Finishing\F1"

# Iterasi melalui setiap folder dataset
for item in os.listdir(parent_directory):
    dataset_path = os.path.join(parent_directory, item)
    if os.path.isdir(dataset_path):
        process_dataset(dataset_path)
