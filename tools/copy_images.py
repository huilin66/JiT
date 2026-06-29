import csv
import shutil
import os

csv_path = r"D:\zhl\data\eccv_dn\sample_check.csv"
source_base = r"D:\zhl\data\eccv_dn\sample_check"
dest_base = source_base

dest_dirs = {
    ("day", 1): os.path.join(dest_base, "day_is_blur"),
    ("day", 0): os.path.join(dest_base, "day_is_not_blur"),
    ("night", 1): os.path.join(dest_base, "night_is_blur"),
    ("night", 0): os.path.join(dest_base, "night_is_not_blur"),
}
for k,v in dest_dirs.items():
    os.makedirs(v, exist_ok=True)
counts = {k: 0 for k in dest_dirs.keys()}

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        time_of_day = row['time_of_day']
        is_blur = int(row['is_blur'])
        filename = row['relative_folder'] + ".png"
        
        key = (time_of_day, is_blur)
        if key not in dest_dirs:
            continue
        
        src_path = os.path.join(source_base, time_of_day, filename)
        dest_path = os.path.join(dest_dirs[key], filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            counts[key] += 1
        else:
            print(f"File not found: {src_path}")

print("Copy summary:")
for (time_of_day, is_blur), count in counts.items():
    blur_str = "is_blur" if is_blur else "is_not_blur"
    print(f"{time_of_day}_{blur_str}: {count} files")