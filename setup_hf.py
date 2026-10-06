from pathlib import Path
from huggingface_hub import HfApi

DATASET_ROOT = Path(
    "/mnt/drive2/Datasets/unitree_g1_data-830-episodes_split"
)

REPO_ID = "rajat-vajpayee/unitree_g1_data-830-episodes_split"


def main():
    train = DATASET_ROOT / "train"
    val = DATASET_ROOT / "val"

    if not train.exists():
        raise FileNotFoundError(f"Missing train directory: {train}")

    if not val.exists():
        raise FileNotFoundError(f"Missing val directory: {val}")

    print(f"Train: {train}")
    print(f"Val:   {val}")

    api = HfApi()

    # Create the dataset repository if it doesn't already exist.
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True,
    )

    # Upload train/
    print("Uploading train...")
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(train),
        path_in_repo="train",
    )

    # Upload val/
    print("Uploading val...")
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(val),
        path_in_repo="val",
    )

    # Verify
    info = api.repo_info(
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    print()
    print("SUCCESS")
    print(f"Repo exists: {info.id}")
    print(f"URL: https://huggingface.co/datasets/{info.id}")


if __name__ == "__main__":
    main()
