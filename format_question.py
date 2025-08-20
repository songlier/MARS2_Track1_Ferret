import json
import os
from PIL import Image

# === 修改这两个路径 ===
input_file = "VG-RS-question.json"         # 输入：你给的原始 JSON 文件
output_file = "VG-RS-refcoco-format.json"  # 输出：转换后的 JSON 文件
image_root = "./images"                  # 图片根目录（与 image_path 中的路径对应）

# === 读取输入 JSON ===
with open(input_file, "r") as f:
    data = json.load(f)

output_data = {
    "images": []
}

used_ids = set()
next_id = 1

for item in data:
    image_path = item["image_path"].replace("\\", "/")
    image_full_path = os.path.join(image_root, os.path.basename(image_path))

    # 自动读取图像尺寸
    try:
        with Image.open(image_full_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"❌ 无法读取图片: {image_full_path}，使用默认尺寸 640x480")
        width, height = 640, 480

    caption = item["question"]
    file_name = os.path.basename(image_path)

    # 分配唯一 ID
    while next_id in used_ids:
        next_id += 1

    output_data["images"].append({
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": next_id,
        "original_id": next_id,
        "caption": caption,
        "dataset_name": "custom",
        "tokens_negative": []
    })

    used_ids.add(next_id)
    next_id += 1

# === 保存输出 JSON ===
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"✅ 转换完成，输出保存为：{output_file}")
