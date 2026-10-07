"""Segment sampling for training and uniform sampling for evaluation."""

import numpy as np


def multi_segments_sampling(clip_length, num_frames):
    average_duration = num_frames // clip_length
    if average_duration > 0:
        return np.arange(clip_length) * average_duration + np.random.randint(
            average_duration, size=clip_length
        )
    return np.clip(np.arange(clip_length), 0, num_frames - 1)


def uniform_sampling(clip_length, num_frames):
    if num_frames > clip_length:
        tick = num_frames / float(clip_length)
        return np.array([int(tick / 2.0 + tick * i) for i in range(clip_length)])
    return np.clip(np.arange(clip_length), 0, num_frames - 1)
