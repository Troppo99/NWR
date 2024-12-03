import os

directory = "D:/NWR/videos/test"
rename = "broom_test_"
index = 1

for file in os.listdir(directory):
    if file.endswith(".mp4"):
        new_name = f"{rename}{str(index).zfill(4)}.mp4"
        os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
        index += 1
