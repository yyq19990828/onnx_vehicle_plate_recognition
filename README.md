# 基于ONNX模型的车辆与车牌识别项目

本项目旨在实现一个完整的交通场景图像处理流程，包括使用ONNX模型进行车辆和车牌的目标检测，并对车牌进行OCR识别。

## 项目特色

- **模型驱动**: 基于ONNX模型进行推理，易于部署和跨平台使用。
- **模块化设计**: 代码结构清晰，分为目标检测、车牌预处理、OCR识别等模块。
- **模拟实现**: 在没有真实模型的情况下，通过模拟函数完整演示了整个处理流程，方便快速验证和后续替换。
- **结构化输出**: 将识别结果以JSON格式输出，同时在原图上进行标注，直观展示结果。

## 项目结构

```
onnx_vehicle_plate_recognition/
├── data/
│   ├── .gitkeep
│   └── sample.jpg  (需要您自行添加一张测试图片)
├── models/
│   └── .gitkeep    (请将您的ONNX模型文件放在这里)
├── runs/
│   └── .gitkeep    (运行结果将保存在这里)
├── main.py         (主程序)
├── requirements.txt(项目依赖)
└── README.md       (项目说明)
```

## 安装

1.  克隆或下载本项目。
2.  安装所需的Python依赖库：

```bash
pip install -r requirements.txt
```

## 使用方法

1.  在 `data/` 目录下放置一张名为 `sample.jpg` 的测试图片。
2.  (可选) 将您的目标检测和OCR ONNX模型文件放入 `models/` 目录。
3.  运行主程序：

```bash
python main.py
```

您也可以通过命令行参数指定输入和输出路径：

```bash
python main.py --input-image /path/to/your/image.jpg --output-dir /path/to/output/folder
```

## 输出结果

程序运行后，将在 `runs/` (或您指定的输出) 目录下生成：

- `result.jpg`: 标注了车辆和车牌识别结果的图片。
- `result.json`: 结构化的识别结果，包含车辆和车牌的位置、类型及号码。
- `plate_*.jpg`: 预处理后的车牌图片，用于调试和检查。

## 注意

当前版本的 `main.py` 使用的是**模拟函数**来代替真实的ONNX模型推理。要使用您自己的模型，您需要修改 `detect_objects` 和 `recognize_plate_ocr` 函数，以加载您的 `.onnx` 文件并执行实际的推理操作。