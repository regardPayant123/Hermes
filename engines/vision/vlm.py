"""Vision-language model adapter using Transformers image-text-to-text pipelines."""

from ._image import load_pil_image, model_id_or_error
from .interface import ImageInput, VLMProvider, VLMResponse


class TransformersVLM(VLMProvider):
    def __init__(self, model_id: str, device: int = -1, max_new_tokens: int = 256):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("Install Transformers with: pip install transformers torch") from exc
            self._pipeline = pipeline(
                "image-text-to-text",
                model=model_id_or_error(self.model_id, "vision-language model"),
                device=self.device,
            )
        return self._pipeline

    async def ask(self, image: ImageInput, prompt: str) -> VLMResponse:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        pil_image = load_pil_image(image)
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }]
        output = self.load()(
            text=conversation,
            max_new_tokens=self.max_new_tokens,
        )
        text = self._extract_text(output).strip()
        if not text:
            raise RuntimeError("VLM returned an empty response")
        return VLMResponse(text=text, model=self.model_id)

    @staticmethod
    def _extract_text(output) -> str:
        if isinstance(output, list) and output:
            output = output[0]
        if isinstance(output, dict):
            generated = output.get("generated_text", output.get("text", ""))
            if isinstance(generated, str):
                return generated
            if isinstance(generated, list) and generated:
                last = generated[-1]
                if isinstance(last, dict):
                    content = last.get("content", "")
                    if isinstance(content, list):
                        return " ".join(
                            item.get("text", "") for item in content if isinstance(item, dict)
                        )
                    return str(content)
                return str(last)
        return str(output)
