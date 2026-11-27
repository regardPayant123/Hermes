# Vision Engine

Vision providers are selected in `configs/vision.json`. Each enabled capability is instantiated by `VisionFactory.from_json`; model weights are loaded lazily on the first inference call.

## Capabilities

- Object detection: Ultralytics YOLO/RT-DETR weights or Hugging Face object-detection models.
- Face detection: OpenCV YuNet ONNX weights or a face-trained Ultralytics checkpoint.
- VLM: Transformers image-text-to-text models such as SmolVLM and Qwen2.5-VL.
- OCR: Tesseract, EasyOCR, or PaddleOCR.
- Optional: image classification, semantic segmentation, human pose estimation, and monocular depth estimation.

Use the model catalog in `engines/vision/catalog.py` to see starter choices. Set `enabled` to `false` for capabilities not needed by the application. For local weights, put their path in `options.model_path`; for Hugging Face models, select a repository ID in `options.model_id`. The `stream.source` setting accepts a camera index, video path, or network stream URL.

Install only the backend needed for each selected provider. Typical packages are `ultralytics`, `transformers` with `torch`, `opencv-contrib-python`, `pytesseract` plus the Tesseract system executable, `easyocr`, or `paddleocr`. Models may download weights on first use when configured with a model name or repository ID; use local paths and the backend's offline settings when downloads are not desired.

Example:

```python
from engines.vision import VisionFactory

vision = VisionFactory.from_json("configs/vision.json")
objects = await vision["object_detection"].detect("sample.jpg")
answer = await vision["vlm"].ask("sample.jpg", "Describe this image.")
text = await vision["ocr"].extract_text("sample.jpg")

async for output in vision["stream"].process(
	vision["object_detection"].detect,
	every_n_frames=2,
):
	print(output.frame_index, output.result)
```

`VisionStream` also accepts an async iterable of image inputs. It reads and processes frames sequentially to bound memory, releases camera/video resources when iteration stops, and can limit or subsample inference with `max_frames` and `every_n_frames`.
