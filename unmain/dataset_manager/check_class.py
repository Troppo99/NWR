import os
import yaml

directory = r"D:\SBHNL\Images\BSML\Datasets\Finishing\F1"
n = 0

for item in os.listdir(directory):
    item_path = os.path.join(directory, item)
    if os.path.isdir(item_path):
        data_yaml_path = os.path.join(item_path, "data.yaml")
        if os.path.isfile(data_yaml_path):
            n += 1
            with open(data_yaml_path, "r") as file:
                data = yaml.safe_load(file)
                names = data["names"]
                print(f"{n}. {item}:\n{names} : <<{len(names)}>>")
        else:
            print(f"data.yaml TIDAK ditemukan di folder {item}")
