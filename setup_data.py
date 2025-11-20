import io
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

dataset_path = snapshot_download(
    repo_id="jinhybr/rvl_cdip_400_train_val_test", repo_type="dataset"
)

train_idx = 0

for file in (Path(dataset_path) / "data").iterdir():
    df = pd.read_parquet(file)
    split = file.stem.split("-")[0]
    target_dir = Path("data") / split

    target_dir.mkdir(parents=True, exist_ok=True)

    list_file = (target_dir / "list.txt").open("a", encoding="utf-8")

    if split == "train":
        train_idx += 1
        train_len = len(df)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=split):
        label = row["label"]
        img_bytes = row["image"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes))
        img_name = f"{idx+train_len if split=='train' and train_idx==2 else idx}.jpg"

        img.save(target_dir / img_name)
        list_file.write(f"{img_name} {label}\n")

    list_file.close()
