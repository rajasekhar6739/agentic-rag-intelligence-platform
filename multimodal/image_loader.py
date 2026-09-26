from pathlib import Path


class ImageLoader:

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp"
    }

    def load(self, file_path: str) -> dict:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {file_path}"
            )

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image type: {path.suffix}"
            )

        return {
            "file_name": path.name,
            "file_path": str(path),
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size
        }