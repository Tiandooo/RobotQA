from pathlib import Path
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from models.img_aug import ImageAugmenter

app = FastAPI()

@app.get("/img_aug")
async def img_aug(image_path: Union[str, Path], save_path: Union[str, Path], num_augments: int = 1):

    image_augmenter: ImageAugmenter = ImageAugmenter(image_path, save_path)
    image_augmenter.apply_augmentations(num_augments)

    return "Augment success!"

