from torch import nn
from typing import Optional, Union, Tuple, Any
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from functools import partial


class FireSegmenter():
    def __init__(self,
                 size: Union[int, Tuple[int, int]] = (1280, 720),
                 pretrained_args: Any = '../weights/segmentation/segformer-b0-segments-flame',
                 ) -> None:
        super().__init__()
        self.size = size
        self.input_fn = partial(SegformerFeatureExtractor(size=size),
                                return_tensors="pt")
        self.predict_fn = partial(SegformerForSemanticSegmentation.from_pretrained(pretrained_args),
                                  return_dict=False)

    def output_fn(self, logits):
        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=self.size,
            mode='bilinear',
            align_corners=False
        )

        # Second, apply argmax on the class dimension
        return upsampled_logits.argmax(dim=1)[0]

    def __call__(self, image):
        inputs = self.input_fn(image)
        predictions = self.predict_fn(inputs)
        outputs = self.output_fn(predictions)

        return outputs
