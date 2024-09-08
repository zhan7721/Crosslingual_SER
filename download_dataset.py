import datasets
from datasets import load_dataset
from utils import map_labels
import os

dataset_match = {"cn-esd-5fold-": "cn_5fold", 
                 "de-pavoque-5fold-": "de_5fold", 
                 "en-iemocap-5fold-": "en_5fold"}
split_list = ['train', 'eval', 'test']

root_path = os.getcwd()

for ds_name, ds_path in dataset_match.items():
    dataset_path = os.path.join(root_path, ds_path)
    os.makedirs(dataset_path, exist_ok=True)

    for i in range(5):
        fold_name = "fold_" + str(i)
        fold_path = os.path.join(dataset_path, fold_name)
        os.makedirs(fold_path, exist_ok=True)

        dataset_name = "zhan7721/" + ds_name + str(i)
        print(f"Loading {dataset_name} to {fold_path}:")
        ds = load_dataset(dataset_name, cache_dir=root_path)
        ds = ds.map(map_labels)
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
        ds.save_to_disk(fold_path)
