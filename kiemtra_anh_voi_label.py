# check_empty_labels.py - Kiểm tra ảnh tương ứng

from pathlib import Path


def check_empty_labels():
    """Kiểm tra xem ảnh empty có tồn tại không"""

    label_dir = Path("/content/test/labels")
    img_dir = Path("/content/test/images")

    empty_labels = []

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            content = f.read().strip()

        # Nếu label rỗng
        if len(content) == 0:
            empty_labels.append(label_file.stem)

    print(f"🔍 Tìm thấy {len(empty_labels)} empty labels\n")

    # Kiểm tra ảnh tương ứng
    orphan_images = []
    existing_images = []

    for label_stem in empty_labels[:10]:  # Kiểm tra 10 cái đầu
        found = False
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            img_file = img_dir / (label_stem + ext)
            if img_file.exists():
                existing_images.append(img_file.name)
                found = True
                break

        if not found:
            orphan_images.append(label_stem)

    print(f"📊 Kiểm tra {len(empty_labels[:10])} empty labels:")
    print(f"   ✅ Có ảnh tương ứng: {len(existing_images)}")
    print(f"   ❌ Không có ảnh:     {len(orphan_images)}")

    if existing_images:
        print(f"\n   Ví dụ ảnh existing: {existing_images[:3]}")
    if orphan_images:
        print(f"\n   Ví dụ ảnh orphan: {orphan_images[:3]}")


if __name__ == "__main__":
    check_empty_labels()
