import json
import os

with open("D:/PycharmProjects/pythonProject1/train_anno.json", 'r') as f:
    data = json.load(f)


for vid in data.keys():

    # num_frames = len(os.listdir(os.path.join("E:\\dataset\\figure_skating\\pair\\pairs\\dataset\\train\\videos",vid)))
    info=data[vid]['annotations']

    print(info[0]['label'])