import sys

from multimodal.image_loader import ImageLoader
from multimodal.vision_processor import VisionProcessor


def main(image_path: str) -> int:

    loader = ImageLoader()

    metadata = loader.load(
        image_path
    )

    print("\n==============================")
    print("IMAGE METADATA")
    print("==============================")

    print(metadata)

    processor = VisionProcessor()

    description = processor.analyze(
        image_path
    )

    print("\n==============================")
    print("VISION OUTPUT")
    print("==============================")

    print(description)

    return 0


def test_multimodal_import():
    """
    Pytest smoke test.

    Actual image processing is performed through main()
    because this module is also a CLI utility.
    """
    assert ImageLoader is not None
    assert VisionProcessor is not None


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: "
            "python -m tests.test_multimodal <image_path>"
        )

        raise SystemExit(1)

    raise SystemExit(
        main(sys.argv[1])
    )