import json
import re
import os

# 配置路径
jsonl_path = "./refexp_result/finetune_refcocog_test/0_of_1.jsonl"
image_dir = "./images"
output_json_path = "VG-RS-answers_ferret_7b_model.json"

# 模型输入尺寸
MODEL_INPUT_SIZE = (1000, 1000)  # width, height


def extract_bbox(text):
    """
    从预测文本中提取 bounding box 坐标，如 [x1, y1, x2, y2]
    """
    standard_match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)
    if standard_match:
        return list(map(int, standard_match.groups()))

    # 标准匹配失败，尝试宽松匹配
    loose_match = re.search(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text)
    if loose_match:
        coords = list(map(int, loose_match.groups()))
        print(f"⚠️ 标准匹配失败但宽松匹配成功：{text}")
        print(f"➡️ 提取到的坐标: {coords}")
        return list(map(int, loose_match.groups()))
    
    # match = re.search(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text)
    # if match:
    #     return list(map(int, match.groups()))
    return None


def scale_bbox_float(bbox, original_size, model_input_size):
    """
    将bbox从模型输入尺寸还原为原图尺寸，保留浮点数
    """
    x1, y1, x2, y2 = bbox
    orig_w, orig_h = original_size
    input_w, input_h = model_input_size

    scale_x = orig_w / input_w
    scale_y = orig_h / input_h

    x1_new = x1 * scale_x
    y1_new = y1 * scale_y
    x2_new = x2 * scale_x
    y2_new = y2 * scale_y

    return [[x1_new, y1_new], [x2_new, y2_new]]


def convert_predictions(jsonl_file, image_folder):
    results = []

    with open(jsonl_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())

            image_file = os.path.join(image_folder, item["file_name"])
            if not os.path.exists(image_file):
                print(f"❌ 图片不存在: {image_file}")
                continue

            bbox = extract_bbox(item["text"])
            if bbox is None:
                print(f"❌ 无法从文本中提取框: {item['file_name']}: {item['text']}")
                continue

            orig_w = item["width"]
            orig_h = item["height"]
            restored_bbox = scale_bbox_float(bbox, (orig_w, orig_h), MODEL_INPUT_SIZE)

            # 构造输出
            output_item = {
                "image_path": os.path.join("images", item["file_name"]).replace("/", "\\"),
                "question": item["prompt"],
                "result": restored_bbox
            }
            results.append(output_item)

    return results


# 执行转换
converted_results = convert_predictions(jsonl_path, image_dir)

# 写入JSON文件
with open(output_json_path, 'w') as f:
    json.dump(converted_results, f, indent=2)

print(f"✅ 转换完成，输出保存至：{output_json_path}")
