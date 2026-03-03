import os
import shutil
from tqdm import tqdm


DATA_DAY = r'/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/DayRainDrop_Train'
DATA_NIGHT = r'/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/NightRainDrop_Train'
DATA_MERGE = r'/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train'
DROP_DIR_NAME = 'Drop'
CLEAR_DIR_NAME = 'Clear'

def copy_data(src_dir, dst_dir, drop_dir_name=DROP_DIR_NAME, clear_dir_name=CLEAR_DIR_NAME, type='Day'):
    """
    Copy data from src_dir to dst_dir.
    """
    dst_dir_drop = os.path.join(dst_dir, drop_dir_name)
    dst_dir_clear = os.path.join(dst_dir, clear_dir_name)
    os.makedirs(dst_dir_drop, exist_ok=True)
    os.makedirs(dst_dir_clear, exist_ok=True)

    src_dir_drop = os.path.join(src_dir, drop_dir_name)
    src_dir_clear = os.path.join(src_dir, clear_dir_name)

    src_drop_list = os.listdir(src_dir_drop)
    for sub_id_name in tqdm(src_drop_list, desc=f'Copy {type} Drop Data'):
        src_id_path = os.path.join(src_dir_drop, sub_id_name)
        for file_name in os.listdir(src_id_path):
            src_file_path = os.path.join(src_id_path, file_name)
            dst_file_path = os.path.join(dst_dir_drop, f'{type}_{sub_id_name}_{file_name}')
            shutil.copy(src_file_path, dst_file_path)
    
    src_clear_list = os.listdir(src_dir_clear)
    for sub_id_name in tqdm(src_clear_list, desc=f'Copy {type} Clear Data'):
        src_id_path = os.path.join(src_dir_clear, sub_id_name)
        for file_name in os.listdir(src_id_path):
            src_file_path = os.path.join(src_id_path, file_name)
            dst_file_path = os.path.join(dst_dir_clear, f'{type}_{sub_id_name}_{file_name}')
            shutil.copy(src_file_path, dst_file_path)


if __name__ == '__main__':
    copy_data(DATA_DAY, DATA_MERGE)
    copy_data(DATA_NIGHT, DATA_MERGE)