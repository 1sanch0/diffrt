from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
import matplotlib.gridspec as gridspec

def plot_images(images: list[tuple[int, np.ndarray]], cols: int):
  images.sort(key=lambda x: int(x[0]))

  reference = images[0]
  images = images[1:]

  rows = ceil(len(images) / cols)
  rows += 1  # Add an extra row for the reference image

  fig, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
  gs = gridspec.GridSpec(rows, cols , figure=fig)

  ax_ref = fig.add_subplot(gs[0, cols // 2])
  ax_ref.axis("off")
  ax_ref.imshow(reference[-1])
  ax_ref.set_title("Reference Image")

  for row in range(rows):
    for col in range(cols):
      ax[row, col].axis("off")

  for i, (num, loss, img) in enumerate(images):
    row = i // cols + 1  # Start from the second row (first is for reference)
    col = i % cols
    ax[row, col].imshow(img)
    ax[row, col].set_title(f"Image {num} " + (f"Loss: {loss:.4f}" if loss is not None else ""))

  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Display images from .ppm files.")
  parser.add_argument("--cols", type=int, default=5, help="Number of columns for the image grid.")
  parser.add_argument("directory", type=str, help="Directory containing the .ppm files.")
  args = parser.parse_args()

  path = Path(args.directory)

  images = []
  for file in path.glob("*.ppm"):
    with open(file, "rb") as f:
      img = Image.open(f)
      split = file.stem.split("_")
      if len(split) < 2:
        num = int(split[-1])
        loss = None
      elif len(split) == 3:
        num = int(split[-1])
        loss = float(split[-2])
      else:
        raise ValueError(f"Unexpected file naming format: {file.stem}")

      # num = int(file.stem.split("_")[-1])
      images.append((num, loss, np.array(img.convert("RGB"))))

  plot_images(images, cols=args.cols)