# 🧭 Intruduction
This repository provides a batch inference pipeline using Ferret, an end-to-end multimodal large language model (MLLM), for Multimodal Reasoning Competition Track1 (VG-RS). Given a set of image-question pairs, the model outputs the corresponding bounding box coordinates through fine-grained referring and grounding.

---
## 📦 Environment Prepare
1. Clone this repository and navigate to FERRET folder
2. Install Package
```Shell
conda create -n ferret python=3.10 -y
conda activate ferret
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install pycocotools
pip install protobuf==3.20.0
```
3. Install additional packages for training cases
```
pip install ninja
pip install flash-attn --no-build-isolation
```

---
## 🗂 Directory Structure

```
project_root/  
├── pyproject. toml			# environment info  
├── model/					# Folder with ferret model  
│   ├── ferret-7b-v1-3  
│   └── ferret-13b-v1-3  
├── images/					# Folder with images   
│   └── `*.jpg` / `*.png`  
├── refexp_result/				# Folder with output  
│   └── finetune_refcocog_test/  
│       └──`0_of_1.jsonl`  
├── download/					# place to unzip the downloaded  
├── VG-RS-question.json			# Input questions and image paths
├── VG-RS-refcoco-format.json			# Input after format  
├── format_question.py				# format input  
├── format_ferret_answer.py			# format answer  
...  
```

---
## 🔧 Model and Processor Setup
### 🧠 Download Model with ModelScope
first download weights of Vicuna following the instructions [here](https://github.com/lm-sys/FastChat#model-weights). Vicuna v1.3 is used in FERRET.  
Then download the prepared offsets of weights: [7B](https://docs-assets.developer.apple.com/ml-research/models/ferret/ferret-7b/ferret-7b-delta.zip), [13B](https://docs-assets.developer.apple.com/ml-research/models/ferret/ferret-13b/ferret-13b-delta.zip) using `wget` or `curl`, and unzip the downloaded offsets to the `download` folder.  
Lastly, apply the offset to the Vicuna's weight by running the following script:  

```Shell
# 7B
python3 -m ferret.model.apply_delta \
    --base ./download/vicuna-7b-v1-3 \
    --target ./model/ferret-7b-v1-3 \
    --delta ./download/ferret-7b-delta
# 13B
python3 -m ferret.model.apply_delta \
    --base ./download/vicuna-13b-v1-3 \
    --target ./model/ferret-13b-v1-3 \
    --delta ./download/ferret-13b-delta
```

---
## 🧪 How to Run Inference
### ✍️ Prepare Input JSON
The file `VG-RS-question.json` should be a list of entries in this format:

```
[
  { 
    "image_path": "images\example.jpg", 
    "question": "What object is next to the red car?"
    },
  ...
]
```

then run the format script

```shell
python format_question.py
```



### 🚀 Run Script
```shell
CUDA_VISIBLE_DEVICES=0 python -m ferret.eval.model_refcoco \
    --model-path ./model/ferret-7b-v1-3 \
    --image_path ./images \
    --data_path ./VG-RS-refcoco-format.json \
    --answers-file refexp_result/finetune_refcocog_test \
    --add_region_feature \
    --chunk-idx 0 \
    --num-chunks 1
CUDA_VISIBLE_DEVICES=0 python -m ferret.eval.model_refcoco \
    --model-path ./model/ferret-13b-v1-3 \
    --image_path ./images \
    --data_path ./VG-RS-refcoco-format.json \
    --answers-file refexp_result/finetune_refcocog_test \
    --add_region_feature \
    --chunk-idx 0 \
    --num-chunks 1
```

---
## 📤 Output Format
```shell
python format_ferret_answer.py
```
The result will be saved as a JSON file containing predicted bounding boxes for each input:

```
[   
  {
    "image_path": "images\example.jpg",     
    "question": "What object is next to the red car?",    
    "result": [[x1, y1], [x2, y2]]     
  },    
  ...     
]
```

> Note: Bounding boxes are in the format `[[x_min, y_min], [x_max, y_max]]`.

---
## 📝 Reference
If you use this code and our data, please cite:
> @article{yao2025lens, title={LENS: Multi-level Evaluation of Multimodal Reasoning with Large Language Models}, author={Yao, Ruilin and Zhang, Bo and Huang, Jirui and Long, Xinwei and Zhang, Yifang and Zou, Tianyu and Wu, Yufei and Su, Shichao and Xu, Yifan and Zeng, Wenxi and others}, journal={arXiv preprint arXiv: 2505.15616}, year={2025} }

> @article{Qwen2VL, title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution}, author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang}, journal={arXiv preprint arXiv: 2409.12191}, year={2024} }

> @article{you2023ferret, title={Ferret: Refer and Ground Anything Anywhere at Any Granularity}, author={You, Haoxuan and Zhang, Haotian and Gan, Zhe and Du, Xianzhi and Zhang, Bowen and Wang, Zirui and Cao, Liangliang and Chang, Shih-Fu and Yang, Yinfei}, journal={arXiv preprint arXiv:2310.07704}, year={2023} }

---
## 🔗 Acknowledgement

This codebase is partially based on [Ferret](https://github.com/apple/ml-ferret)

## 💬 Contact

If you encounter any issues or have questions, feel free to open an issue on GitHub.
