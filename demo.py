import json
import os

from tqdm import tqdm

with open("/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2/Drop_scen_pred.json", "r") as f:
    scene_info = json.load(f)

file_list = os.listdir(r'/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2/Drop')

for file in tqdm(file_list):
    if file=='Day_00051_00054.png':
        print(file)