import os
import urllib.request
import zipfile


ZIP_FILES = [
    "images-20251223T085919Z-3-001.zip",
    "images-20251223T085919Z-3-002.zip",
    "images-20251223T085919Z-3-003.zip",
    "images-20251223T085919Z-3-004.zip",
    "images-20251223T085919Z-3-005.zip",
    "images-20251223T085919Z-3-006.zip",
    "images-20251223T085919Z-3-007.zip",
    "images-20251223T085919Z-3-008.zip",
]


def _marker_dir(images_dir: str) -> str:
    return os.path.join(images_dir, ".extracted")


def _marker_path(images_dir: str, zip_name: str) -> str:
    return os.path.join(_marker_dir(images_dir), f"{zip_name}.done")


def _is_complete(images_dir: str, zip_name: str) -> bool:
    return os.path.exists(_marker_path(images_dir, zip_name))


def _mark_complete(images_dir: str, zip_name: str) -> None:
    os.makedirs(_marker_dir(images_dir), exist_ok=True)
    with open(_marker_path(images_dir, zip_name), "w", encoding="utf-8") as f:
        f.write("ok\n")


def download_file(url: str, dest_path: str) -> None:
    """Download a single zip file with minimal logging."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        if zipfile.is_zipfile(dest_path):
            print(f"Using cached archive: {os.path.basename(dest_path)}")
            return
        print(f"Cached archive is invalid, re-downloading: {os.path.basename(dest_path)}")
        os.remove(dest_path)

    print(f"Downloading: {os.path.basename(dest_path)}")
    urllib.request.urlretrieve(url, dest_path)


def unzip_file(zip_path: str, dest_dir: str, zip_name: str) -> None:
    """Extract a zip file once and persist a completion marker."""
    if _is_complete(dest_dir, zip_name):
        print(f"Already extracted, skipping: {zip_name}")
        return
    print(f"Extracting: {zip_name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    _mark_complete(dest_dir, zip_name)


def main() -> int:
    """Download and extract all image archives into Data/images/."""
    # Use current working directory and create/find Data folder there
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "Data")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    base_url = "https://storage.googleapis.com/unido-afririce/"
    for zip_name in ZIP_FILES:
        url = f"{base_url}{zip_name}"
        zip_path = os.path.join(data_dir, zip_name)

        if not _is_complete(images_dir, zip_name):
            download_file(url, zip_path)
            unzip_file(zip_path, images_dir, zip_name)
            if os.path.exists(zip_path):
                os.remove(zip_path)
        else:
            print(f"Archive complete marker found, skipping: {zip_name}")

    print("All downloads and extractions completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
