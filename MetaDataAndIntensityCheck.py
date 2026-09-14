from PIL import Image, ExifTags
import numpy as np, glob, os

MENDELEY_ROOT = r"D:\Research Work\TB Classification\Mendeley Dataset\Mendeley Data"

for cls in ["Normal Chest X-rays", "TB Chest X-rays"]:
    folder = os.path.join(MENDELEY_ROOT, cls)
    files = sorted(glob.glob(os.path.join(folder, "*")))
    print(f"\n=== {cls} ===  n={len(files)}")

    exts = {}
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        exts[ext] = exts.get(ext, 0) + 1
    print("extensions:", exts)

    sample = files[:80]
    sizes, means, stds, modes, filesizes = [], [], [], [], []
    for f in sample:
        img = Image.open(f)
        modes.append(img.mode)
        arr = np.array(img.convert("L"), dtype=np.float64)
        sizes.append(arr.shape)
        means.append(arr.mean())
        stds.append(arr.std())
        filesizes.append(os.path.getsize(f))

    print("image modes:", set(modes))
    print("unique sizes (up to 10):", sorted(set(sizes))[:10])
    print(f"mean intensity: {np.mean(means):.2f} ± {np.std(means):.2f}")
    print(f"mean std-dev:   {np.mean(stds):.2f} ± {np.std(stds):.2f}")
    print(f"file size KB:   {np.mean(filesizes)/1024:.1f} ± {np.std(filesizes)/1024:.1f}")