import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

dataset_path = Path("dataset")

normal_folder = dataset_path / "train" / "NORMAL"
pneumonia_folder = dataset_path / "train" / "PNEUMONIA"

normal_image = next(normal_folder.glob("*"))
pneumonia_image = next(pneumonia_folder.glob("*"))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(Image.open(normal_image), cmap="gray")
axes[0].set_title("NORMAL")
axes[0].axis("off")

axes[1].imshow(Image.open(pneumonia_image), cmap="gray")
axes[1].set_title("PNEUMONIA")
axes[1].axis("off")

plt.tight_layout()
plt.show()