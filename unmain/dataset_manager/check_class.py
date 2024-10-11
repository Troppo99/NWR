import os
import yaml


def replace_characters(names_list, replace_dash):
    if replace_dash:
        return [name.replace("-", " ") for name in names_list]
    else:
        return names_list


directory = r"D:\SBHNL\Images\BSML\Datasets\Finishing\F1"
n = 0
replace_dash_with_space = False

for item in os.listdir(directory):
    item_path = os.path.join(directory, item)
    if os.path.isdir(item_path):
        data_yaml_path = os.path.join(item_path, "data.yaml")
        if os.path.isfile(data_yaml_path):
            n += 1
            with open(data_yaml_path, "r") as file:
                data = yaml.safe_load(file)
                names = data.get("names", [])
                modified_names = replace_characters(names, replace_dash_with_space)
                data["names"] = modified_names
            with open(data_yaml_path, "w") as file:
                yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
            print(f"{n}. {item}:\n{modified_names} : << {len(modified_names)} >>")
        else:
            print(f"data.yaml TIDAK ditemukan di folder {item}")
