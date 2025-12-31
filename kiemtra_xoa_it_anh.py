import os
import yaml
from pathlib import Path
from collections import defaultdict


class SimpleDatasetCleanupPipeline:
    """Pipeline chia 3 bước: 1.Xem số ảnh → 2.Chọn class xóa → 3.Cập nhật YAML"""

    def __init__(self, yaml_path, data_dirs):
        self.yaml_path = yaml_path
        self.data_dirs = data_dirs
        self.class_names = {}
        self.class_counts = {}
        self.class_to_remove = None
        self.images_to_remove = defaultdict(list)
        self.labels_to_modify = defaultdict(list)
        self.stats = {
            "total_images": 0,
            "images_with_class": 0,
            "images_only_class": 0,
            "labels_modified": 0,
        }

    # ==================== BƯỚC 1: XEM DANH SÁCH CLASS VÀ SỐ ẢNH ====================
    def step1_view_classes_and_counts(self):
        """BƯỚC 1️⃣ : Hiển thị danh sách class + số ảnh của mỗi class"""
        print("\n" + "=" * 70)
        print("BƯỚC 1️⃣  : XEM DANH SÁCH CLASS VÀ SỐ ẢNH")
        print("=" * 70)

        # Đọc data.yaml
        print(f"\n📖 Đọc {self.yaml_path}...")
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if "names" not in data:
                print("❌ Không tìm thấy 'names' trong data.yaml")
                return False

            names_data = data["names"]

            # Convert to dict nếu là list
            if isinstance(names_data, list):
                self.class_names = {idx: name for idx, name in enumerate(names_data)}
            elif isinstance(names_data, dict):
                self.class_names = names_data
            else:
                print(f"❌ Format 'names' không hợp lệ: {type(names_data)}")
                return False

        except Exception as e:
            print(f"❌ Lỗi đọc data.yaml: {e}")
            return False

        # Thống kê số ảnh của mỗi class
        print(f"\n📊 Đang thống kê số ảnh của mỗi class...")
        self.class_counts = self._count_images_per_class()

        # Hiển thị kết quả
        print(f"\n✅ Danh sách tất cả classes:")
        print("-" * 70)
        print(f"{'Index':<8} {'Class Name':<30} {'Số ảnh':<10}")
        print("-" * 70)

        for idx in sorted(self.class_names.keys()):
            name = self.class_names[idx]
            count = self.class_counts.get(idx, 0)
            print(f"{idx:<8} {name:<30} {count:<10}")

        print("-" * 70)
        print(
            f"Tổng cộng: {len(self.class_names)} classes, "
            f"{sum(self.class_counts.values())} ảnh (lưu ý: 1 ảnh có thể chứa nhiều class)"
        )
        print("=" * 70)

        return True

    def _count_images_per_class(self):
        """Đếm số ảnh có chứa mỗi class"""
        class_counts = defaultdict(int)

        for split_name, split_info in self.data_dirs.items():
            label_dir = split_info["labels"]

            if not os.path.exists(label_dir):
                continue

            label_files = list(Path(label_dir).glob("*.txt"))

            for label_file in label_files:
                with open(label_file, "r") as f:
                    lines = f.readlines()

                # Tìm tất cả class trong file này
                classes_in_file = set()
                for line in lines:
                    try:
                        class_id = int(line.split()[0])
                        classes_in_file.add(class_id)
                    except (ValueError, IndexError):
                        continue

                # Cộng tổng
                for class_id in classes_in_file:
                    class_counts[class_id] += 1

        return class_counts

    # ==================== BƯỚC 2: CHỌN CLASS VÀ XEM CHI TIẾT ====================
    def step2_select_class(self):
        """BƯỚC 2️⃣ : Chọn class cần xóa"""
        print("\n" + "=" * 70)
        print("BƯỚC 2️⃣  : CHỌN CLASS CẦN XÓA")
        print("=" * 70)

        # Nhập class index
        while True:
            try:
                print(
                    f"\n📝 Nhập thứ tự (index) của class cần xóa (0-{len(self.class_names)-1}): ",
                    end="",
                )
                class_idx = int(input().strip())

                if class_idx not in self.class_names:
                    print(f"❌ Index {class_idx} không tồn tại. Vui lòng nhập lại.")
                    continue

                self.class_to_remove = class_idx
                break

            except ValueError:
                print("❌ Vui lòng nhập số nguyên hợp lệ")

        # Scan dataset với class được chọn
        print(f"\n🔍 Scanning dataset để tìm class {self.class_to_remove}...")
        self.scan_dataset()

        # Hiển thị chi tiết
        print(f"\n✅ CHI TIẾT CLASS ĐƯỢC CHỌN:")
        print("-" * 70)
        print(f"Class ID: {self.class_to_remove}")
        print(f"Class Name: '{self.class_names[self.class_to_remove]}'")
        print(f"Số ảnh chứa class này: {self.stats['images_with_class']}")
        print(
            f"  → Ảnh chỉ có class này (sẽ XÓA toàn bộ): {self.stats['images_only_class']}"
        )
        print(
            f"  → Ảnh có class khác (chỉ modify label): {len([v for vals in self.labels_to_modify.values() for v in vals])}"
        )
        print("-" * 70)

        return True

    def scan_dataset(self):
        """Scan dataset để tìm class cần xóa"""
        self.images_to_remove.clear()
        self.labels_to_modify.clear()
        self.stats = {
            "total_images": 0,
            "images_with_class": 0,
            "images_only_class": 0,
            "labels_modified": 0,
        }

        for split_name, split_info in self.data_dirs.items():
            label_dir = split_info["labels"]
            image_dir = split_info["images"]

            if not os.path.exists(label_dir):
                continue

            label_files = list(Path(label_dir).glob("*.txt"))

            for label_file in label_files:
                with open(label_file, "r") as f:
                    lines = f.readlines()

                has_target_class = False
                has_other_class = False

                for line in lines:
                    try:
                        class_id = int(line.split()[0])
                        if class_id == self.class_to_remove:
                            has_target_class = True
                        else:
                            has_other_class = True
                    except (ValueError, IndexError):
                        continue

                if has_target_class:
                    self.stats["images_with_class"] += 1
                    img_path = Path(image_dir) / label_file.stem

                    if has_other_class:
                        self.labels_to_modify[split_name].append(
                            {
                                "label_file": label_file,
                                "image_file": img_path,
                                "image_dir": image_dir,
                            }
                        )
                    else:
                        self.images_to_remove[split_name].append(
                            {
                                "label_file": label_file,
                                "image_file": img_path,
                                "image_dir": image_dir,
                            }
                        )
                        self.stats["images_only_class"] += 1

                self.stats["total_images"] += 1

    # ==================== BƯỚC 3: XÁC NHẬN VÀ THỰC HIỆN ====================
    def step3_confirm_and_delete(self):
        """BƯỚC 3️⃣ : Xác nhận, xóa & cập nhật YAML"""
        print("\n" + "=" * 70)
        print("BƯỚC 3️⃣  : XÁC NHẬN & THỰC HIỆN XÓA + CẬP NHẬT YAML")
        print("=" * 70)

        # Xác nhận trước khi xóa
        print(f"\n⚠️  XÁC NHẬN XÓA:")
        print(
            f"   Class: [{self.class_to_remove}] = '{self.class_names[self.class_to_remove]}'"
        )
        print(f"   • Xóa {self.stats['images_only_class']} ảnh toàn bộ")
        print(
            f"   • Modify {len([v for vals in self.labels_to_modify.values() for v in vals])} file label"
        )
        print(
            f"   • Cập nhật data.yaml (giảm nc từ {len(self.class_names)} xuống {len(self.class_names) - 1})"
        )

        print(f"\n📝 Xác nhận? (y/n): ", end="")
        if input().strip().lower() != "y":
            print("❌ Hủy bỏ")
            return False

        # Thực hiện xóa
        print("\n🔥 Bắt đầu XÓA...")
        self.execute_deletion()

        # Cập nhật YAML
        print("\n📝 CẬP NHẬT data.yaml...")
        self.update_yaml()

        return True

    def execute_deletion(self):
        """Thực hiện xóa ảnh và cập nhật labels"""
        deleted_images = 0
        deleted_labels = 0
        modified_labels = 0

        # Xóa ảnh + labels
        for split_name, items in self.images_to_remove.items():
            for item in items:
                label_file = item["label_file"]
                image_file = item["image_file"]
                image_dir = item["image_dir"]

                # Xóa ảnh
                for ext in [".jpg", ".png", ".JPG", ".PNG", ".jpeg", ".JPEG"]:
                    img_path = Path(image_dir) / (image_file.name + ext)
                    if img_path.exists():
                        os.remove(img_path)
                        deleted_images += 1
                        break

                # Xóa label
                if label_file.exists():
                    os.remove(label_file)
                    deleted_labels += 1

        # Cập nhật labels (xóa dòng class khỏi files)
        for split_name, items in self.labels_to_modify.items():
            for item in items:
                label_file = item["label_file"]

                with open(label_file, "r") as f:
                    lines = f.readlines()

                filtered_lines = [
                    line
                    for line in lines
                    if int(line.split()[0]) != self.class_to_remove
                ]

                with open(label_file, "w") as f:
                    f.writelines(filtered_lines)

                modified_labels += 1

        # Điều chỉnh class ID (class > removed_class thì -1)
        for split_name, split_info in self.data_dirs.items():
            label_dir = split_info["labels"]
            if os.path.exists(label_dir):
                for label_file in Path(label_dir).glob("*.txt"):
                    with open(label_file, "r") as f:
                        lines = f.readlines()

                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        try:
                            class_id = int(parts[0])
                            if class_id > self.class_to_remove:
                                parts[0] = str(class_id - 1)
                            new_lines.append(" ".join(parts) + "\n")
                        except (ValueError, IndexError):
                            new_lines.append(line)

                    with open(label_file, "w") as f:
                        f.writelines(new_lines)

        print(f"\n✅ HOÀN THÀNH DELETE:")
        print(f"   • Xóa {deleted_images} ảnh")
        print(f"   • Xóa {deleted_labels} file labels")
        print(f"   • Cập nhật {modified_labels} file labels")
        print(f"   • Điều chỉnh class ID trong tất cả labels")

    def update_yaml(self):
        """Cập nhật data.yaml"""
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        old_nc = len(self.class_names)

        # Giảm nc
        data["nc"] = old_nc - 1

        # Xóa class name
        if isinstance(data["names"], list):
            new_names = [
                name
                for idx, name in enumerate(data["names"])
                if idx != self.class_to_remove
            ]
            data["names"] = new_names
        elif isinstance(data["names"], dict):
            new_names = {}
            new_idx = 0
            for idx in sorted(data["names"].keys()):
                if idx != self.class_to_remove:
                    new_names[new_idx] = data["names"][idx]
                    new_idx += 1
            data["names"] = new_names

        # Ghi lại
        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        print(f"   Trước: nc={old_nc}")
        print(f"   Sau:   nc={data['nc']}")
        print(f"\n✅ Đã cập nhật {self.yaml_path}")

    def run_pipeline(self):
        """Chạy 3 bước"""
        print("\n" + "🚀" * 35)
        print("DATASET CLEANUP PIPELINE - 3 BƯỚC (KHÔNG PREVIEW ẢNH)")
        print("🚀" * 35)

        # BƯỚC 1
        if not self.step1_view_classes_and_counts():
            return False

        input("\n➡️  Nhấn ENTER để chuyển sang BƯỚC 2...")

        # BƯỚC 2
        if not self.step2_select_class():
            return False

        input("\n➡️  Nhấn ENTER để chuyển sang BƯỚC 3...")

        # BƯỚC 3
        if not self.step3_confirm_and_delete():
            return False

        print("\n" + "=" * 70)
        print("🎉 HOÀN TẤT! Dataset đã sạch và sẵn sàng train")
        print("=" * 70)
        return True


# ===== CHẠY NGAY =====
if __name__ == "__main__":
    yaml_path = "/content/data_test.yaml"

    data_dirs = {
        "train": {"labels": "/content/train/labels", "images": "/content/train/images"},
        "val": {"labels": "/content/valid/labels", "images": "/content/valid/images"},
        "test": {"labels": "/content/test/labels", "images": "/content/test/images"},
    }

    # Kiểm tra file tồn tại
    if not os.path.exists(yaml_path):
        print(f"❌ File không tồn tại: {yaml_path}")
        exit(1)

    # Chạy pipeline
    pipeline = SimpleDatasetCleanupPipeline(yaml_path, data_dirs)
    pipeline.run_pipeline()
