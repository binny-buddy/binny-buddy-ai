import base64
import os
import random

from app.models.asset import AssetFile, AssetType
from app.models.detection import PlasticType


def read_asset_file(asset_file_path) -> AssetFile:
    """
    Read asset file and return AssetFile object
    :param asset_file_path: path to the asset file
    :return: AssetFile object
    """
    with open(asset_file_path, "rb") as f:
        file_data = f.read()

    file_name = os.path.basename(asset_file_path)
    file_size = os.path.getsize(asset_file_path)

    return AssetFile(
        filename=file_name,
        content_base64=base64.b64encode(file_data).decode("utf-8"),
        size=file_size,
    )


def get_created_asset(
    model: PlasticType,
    asset_type: AssetType,
    asset_dir="app/assets/created",
) -> AssetFile | None:

    # check if asset directory exists
    if not os.path.exists(asset_dir):
        return None
    # check if asset directory is empty
    asset_files = os.listdir(asset_dir)

    # file naming convection {model}_{asset_type}_{datetime.now()}.jpg
    # filter target files
    asset_files = [
        f
        for f in asset_files
        if f.startswith(f"{model.value}_{asset_type.value}") and f.endswith(".jpg")
    ]
    # get random file
    asset_file = random.choice(asset_files)

    return read_asset_file(os.path.join(asset_dir, asset_file))
