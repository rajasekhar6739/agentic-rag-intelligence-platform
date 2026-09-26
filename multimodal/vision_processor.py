from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


class VisionProcessor:

    def __init__(self, model=None, max_tokens=256):

        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(
            api_key=api_key
        )

        # FORCE currently available vision model.
        self.model = (
            model
            or os.getenv("GROQ_VISION_MODEL")
            or "qwen/qwen3.8-27b"
        )

        self.max_tokens = min(
            int(max_tokens),
            256
        )

    @staticmethod
    def _encode_image(image_path):

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        mime_type, _ = mimetypes.guess_type(
            str(path)
        )

        if not mime_type:
            mime_type = "image/png"

        with open(path, "rb") as image_file:

            encoded = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return (
            f"data:{mime_type};base64,{encoded}"
        )

    def analyze(
        self,
        image_path,
        prompt=(
            "Describe this image briefly. "
            "Identify the important objects, "
            "text, and visual information."
        ),
    ):

        image_url = self._encode_image(
            image_path
        )

        response = (
            self.client.chat.completions.create(
                model="qwen/qwen3.8-27b",

                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": str(prompt),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                },
                            },
                        ],
                    }
                ],

                max_completion_tokens=256,
                temperature=0.2,
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
            or ""
        )