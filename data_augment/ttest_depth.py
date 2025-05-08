from PIL import ImageEnhance
from PIL import Image

img_path = "../data/00000.png"
original_image = Image.open(img_path)
augmented_image = original_image.copy()

enhancer = ImageEnhance.Brightness(augmented_image)
augmented_image = enhancer.enhance(1.2)

augmented_image.save("../data/00000_aug.png")