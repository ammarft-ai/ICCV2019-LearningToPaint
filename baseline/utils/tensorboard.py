import PIL
import scipy.misc
from io import BytesIO
import numpy as np
import tensorboardX as tb
from PIL import Image
from tensorboardX.summary import Summary

class TensorBoard(object):
    def __init__(self, model_dir):
        self.summary_writer = tb.FileWriter(model_dir)

    def add_image(self, tag, img, step):
        summary = Summary()
        bio = BytesIO()

        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, Image.Image):
            pass
        else:
            # Convert numpy array to uint8 if necessary
            if img.dtype != np.uint8:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            if img.ndim == 2:  # grayscale
                img = Image.fromarray(img, mode='L')
            elif img.ndim == 3:
                if img.shape[0] in [1, 3] and img.shape[0] != img.shape[-1]:
                    # convert CHW -> HWC if necessary
                    img = np.transpose(img, (1, 2, 0))
                img = Image.fromarray(img)
            else:
                raise ValueError("Unsupported image format")

        img.save(bio, format="PNG")
        image_summary = Summary.Image(encoded_image_string=bio.getvalue())
        summary.value.add(tag=tag, image=image_summary)
        self.summary_writer.add_summary(summary, global_step=step)

    def add_scalar(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, global_step=step)
