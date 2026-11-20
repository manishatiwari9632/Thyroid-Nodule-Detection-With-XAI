
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


class ImageCollectionError(Exception):
    """Raised when image collection cannot be completed."""
    pass


def collect_image_paths(image_paths):
    """
    Collect and validate individually selected image paths.

    Args:
        image_paths: List of image file paths.

    Returns:
        dict:
            valid_images: List of valid image paths.
            invalid_images: List of invalid paths with reasons.
    """

    if not isinstance(image_paths, (list, tuple)):
        raise TypeError("image_paths must be a list or tuple.")

    valid_images = []
    invalid_images = []
    seen_paths = set()

    for image_path in image_paths:

        if not isinstance(image_path, (str, Path)):
            invalid_images.append({
                "path": str(image_path),
                "reason": "Invalid path type."
            })
            continue

        path = Path(image_path).expanduser()

        try:
            resolved_path = path.resolve(strict=True)
        except (OSError, RuntimeError):
            invalid_images.append({
                "path": str(path),
                "reason": "File does not exist or cannot be accessed."
            })
            continue

        if not resolved_path.is_file():
            invalid_images.append({
                "path": str(path),
                "reason": "Path is not a file."
            })
            continue

        if resolved_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            invalid_images.append({
                "path": str(path),
                "reason": "Unsupported image format."
            })
            continue

        if resolved_path in seen_paths:
            continue

        seen_paths.add(resolved_path)
        valid_images.append(resolved_path)

    return {
        "valid_images": valid_images,
        "invalid_images": invalid_images,
    }


def collect_images_from_folder(folder_path):
    """
    Collect supported image files directly from a folder's root.

    Nested subdirectories are not searched.

    Args:
        folder_path: Path to the selected folder.

    Returns:
        dict:
            valid_images: List of valid image paths.
            invalid_images: List of image files that could not be accessed.
    """

    if not isinstance(folder_path, (str, Path)):
        raise TypeError("folder_path must be a string or Path.")

    folder = Path(folder_path).expanduser()

    if not folder.exists():
        raise ImageCollectionError(
            f"Folder does not exist: {folder}"
        )

    if not folder.is_dir():
        raise ImageCollectionError(
            f"Provided path is not a directory: {folder}"
        )

    valid_images = []
    invalid_images = []

    try:
        entries = sorted(
            folder.iterdir(),
            key=lambda path: path.name.lower()
        )
    except OSError as exc:
        raise ImageCollectionError(
            f"Unable to access folder: {folder}. Reason: {exc}"
        ) from exc

    seen_paths = set()

    for path in entries:

        try:
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            resolved_path = path.resolve(strict=True)

            if resolved_path in seen_paths:
                continue

            seen_paths.add(resolved_path)
            valid_images.append(resolved_path)

        except (OSError, RuntimeError) as exc:
            invalid_images.append({
                "path": str(path),
                "reason": f"Unable to access file: {exc}"
            })

    return {
        "valid_images": valid_images,
        "invalid_images": invalid_images,
    }
