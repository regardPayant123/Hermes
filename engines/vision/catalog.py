"""Selectable model choices grouped by computer vision capability."""

MODEL_CATALOG = {
    "object_detection": {
        "ultralytics": [
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "rtdetr-l.pt",
        ],
        "transformers": [
            "hustvl/yolos-tiny",
            "facebook/detr-resnet-50",
            "facebook/detr-resnet-101",
            "IDEA-Research/grounding-dino-tiny",
        ],
    },
    "face_detection": {
        "opencv_yunet": ["face_detection_yunet_2023mar.onnx"],
        "ultralytics": ["custom face detector weights (model_path)"],
    },
    "vlm": {
        "transformers": [
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "llava-hf/llava-1.5-7b-hf",
        ],
    },
    "ocr": {
        "pytesseract": ["installed Tesseract language packs"],
        "easyocr": ["en", "fr", "many supported languages"],
        "paddleocr": ["PaddleOCR PP-OCRv4", "PaddleOCR PP-OCRv5"],
    },
    "image_classification": {
        "transformers": [
            "google/mobilenet_v2_1.0_224",
            "microsoft/resnet-50",
            "google/vit-base-patch16-224",
        ],
    },
    "segmentation": {
        "transformers": [
            "nvidia/segformer-b0-finetuned-ade-512-512",
            "facebook/mask2former-swin-tiny-coco-instance",
        ],
        "ultralytics": ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt"],
    },
    "pose_estimation": {
        "ultralytics": ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt"],
    },
    "depth_estimation": {
        "transformers": [
            "LiheYoung/depth-anything-small-hf",
            "depth-anything/Depth-Anything-V2-Base-hf",
        ],
    },
}
