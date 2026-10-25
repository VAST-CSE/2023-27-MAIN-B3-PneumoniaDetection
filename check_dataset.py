from pathlib import Path

dataset_path = Path("dataset")

for split in ["train", "val", "test"]:
    print(f"\nChecking {split} folder:")

    for category in ["NORMAL", "PNEUMONIA"]:
        folder = dataset_path / split / category

        if folder.exists():
            images = [
                file for file in folder.iterdir()
                if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
            print(f"{category}: {len(images)} images")
        else:
            print(f"{category}: Folder not found!")