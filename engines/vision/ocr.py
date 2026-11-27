"""OCR adapters for Tesseract, EasyOCR, and PaddleOCR."""

from ._image import load_pil_image
from .interface import BoundingBox, Detection, ImageInput, OCRProvider, OCRResult


class TesseractOCR(OCRProvider):
    def __init__(self, language: str = "eng+fra", min_confidence: float = 0.0):
        if not 0.0 <= min_confidence <= 100.0:
            raise ValueError("min_confidence must be between 0 and 100")
        self.language = language
        self.min_confidence = min_confidence

    async def extract_text(self, image: ImageInput) -> OCRResult:
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError as exc:
            raise RuntimeError("Install Tesseract bindings with: pip install pytesseract") from exc

        data = pytesseract.image_to_data(
            load_pil_image(image),
            lang=self.language,
            output_type=Output.DICT,
        )
        words = []
        detections = []
        scores = []
        for index, value in enumerate(data["text"]):
            text = value.strip()
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                continue
            if not text or confidence < self.min_confidence or confidence < 0:
                continue
            x = float(data["left"][index])
            y = float(data["top"][index])
            width = float(data["width"][index])
            height = float(data["height"][index])
            words.append(text)
            scores.append(confidence / 100.0)
            detections.append(Detection(
                label=text,
                confidence=min(1.0, confidence / 100.0),
                box=BoundingBox(x, y, x + width, y + height),
            ))
        return OCRResult(
            text=" ".join(words),
            confidence=sum(scores) / len(scores) if scores else None,
            boxes=tuple(detections),
        )


class EasyOCRProvider(OCRProvider):
    def __init__(self, languages: list[str] | None = None, gpu: bool = False):
        self.languages = languages or ["en", "fr"]
        self.gpu = gpu
        self._reader = None

    def load(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError("Install EasyOCR with: pip install easyocr") from exc
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    async def extract_text(self, image: ImageInput) -> OCRResult:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install NumPy with: pip install numpy") from exc
        results = self.load().readtext(np.asarray(load_pil_image(image)), detail=1)
        boxes = []
        for polygon, text, confidence in results:
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            boxes.append(Detection(
                label=str(text),
                confidence=float(confidence),
                box=BoundingBox(min(xs), min(ys), max(xs), max(ys)),
            ))
        return OCRResult(
            text=" ".join(item.label for item in boxes),
            confidence=sum(item.confidence for item in boxes) / len(boxes) if boxes else None,
            boxes=tuple(boxes),
        )


class PaddleOCRProvider(OCRProvider):
    def __init__(self, language: str = "en"):
        self.language = language
        self._engine = None

    def load(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError("Install PaddleOCR with: pip install paddleocr") from exc
            self._engine = PaddleOCR(lang=self.language)
        return self._engine

    async def extract_text(self, image: ImageInput) -> OCRResult:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install NumPy with: pip install numpy") from exc
        pixels = np.asarray(load_pil_image(image))
        engine = self.load()
        if hasattr(engine, "predict"):
            output = engine.predict(pixels)
            return self._parse_modern(output)
        output = engine.ocr(pixels, cls=True)
        detections = []
        for page in output or []:
            for item in page or []:
                polygon, recognition = item
                text, confidence = recognition
                xs = [float(point[0]) for point in polygon]
                ys = [float(point[1]) for point in polygon]
                detections.append(Detection(
                    label=str(text),
                    confidence=float(confidence),
                    box=BoundingBox(min(xs), min(ys), max(xs), max(ys)),
                ))
        return OCRResult(
            text=" ".join(item.label for item in detections),
            confidence=sum(item.confidence for item in detections) / len(detections) if detections else None,
            boxes=tuple(detections),
        )

    @staticmethod
    def _parse_modern(output) -> OCRResult:
        detections = []
        for page in output or []:
            page = page if isinstance(page, dict) else getattr(page, "json", {})
            if callable(page):
                page = page()
            result = page.get("res", page) if isinstance(page, dict) else {}
            texts = result.get("rec_texts", [])
            scores = result.get("rec_scores", [])
            polygons = result.get("rec_polys", result.get("dt_polys", []))
            for text, confidence, polygon in zip(texts, scores, polygons):
                xs = [float(point[0]) for point in polygon]
                ys = [float(point[1]) for point in polygon]
                detections.append(Detection(
                    label=str(text),
                    confidence=float(confidence),
                    box=BoundingBox(min(xs), min(ys), max(xs), max(ys)),
                ))
        return OCRResult(
            text=" ".join(item.label for item in detections),
            confidence=sum(item.confidence for item in detections) / len(detections) if detections else None,
            boxes=tuple(detections),
        )
