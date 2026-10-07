"""PyAV decoding and CLIP preprocessing for sampled video frames."""

import av
from torchvision.transforms import CenterCrop, Compose

from .sampling import multi_segments_sampling, uniform_sampling
from .transforms import GroupToTensorBCHW, TensorNormalize


class RawVideoExtractorpyAV:
    def __init__(self, size=224, is_train=True, num_segments=32):
        self.train = is_train
        self.num_segments = num_segments
        self.transform = Compose(
            [
                GroupToTensorBCHW(),
                CenterCrop(size),
                TensorNormalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def get_video_data(self, video_path):
        with av.open(video_path) as container:
            frames = list(container.decode(video=0))
            reported_frames = container.streams.video[0].frames
            num_frames = (
                min(reported_frames, len(frames))
                if reported_frames > 0
                else len(frames)
            )
            if not num_frames:
                raise ValueError(f"Video has no decodable frames: {video_path}")
            sample = multi_segments_sampling if self.train else uniform_sampling
            indices = sample(self.num_segments, num_frames)
            images = [frames[i].to_rgb().to_ndarray() for i in indices]
        return self.transform(images), min(num_frames, self.num_segments)
