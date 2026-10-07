"""Video and text datasets for ITA."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .decode import RawVideoExtractorpyAV
from .utils import load_jsonl


def _video_path(video_dir, name, datatype):
    if datatype == "activity" and name.startswith("v_"):
        name = name[2:]
    return str(Path(video_dir) / f"{name}.mp4")


def _tokenize(text, tokenizer, max_words):
    words = tokenizer.tokenize(text)[: max_words - 2]
    words = ["<|startoftext|>"] + words + ["<|endoftext|>"]
    return torch.tensor(tokenizer.convert_tokens_to_ids(words), dtype=torch.long)


class Val_VideoDataset(Dataset):
    def __init__(
        self, jsonl_path, video_dir, max_frames=32, image_resolution=224, datatype=None
    ):
        data = load_jsonl(jsonl_path)
        self.video_names = list(dict.fromkeys(item["vid_name"] for item in data))
        self.video_dir = video_dir
        self.datatype = datatype
        self.rawVideoExtractor = RawVideoExtractorpyAV(
            size=image_resolution, is_train=False, num_segments=max_frames
        )

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        name = self.video_names[idx]
        video, _ = self.rawVideoExtractor.get_video_data(
            _video_path(self.video_dir, name, self.datatype)
        )
        return name, video


class Val_TextDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_words=64):
        self.data = load_jsonl(jsonl_path)
        self.txt2vid = {item["desc_id"]: item["vid_name"] for item in self.data}
        self.tokenizer = tokenizer
        self.max_words = max_words

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["desc_id"], _tokenize(item["desc"], self.tokenizer, self.max_words)


class TrainDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        video_dir,
        tokenizer,
        max_words=64,
        max_frames=32,
        image_resolution=224,
        datatype=None,
    ):
        self.video_dir = video_dir
        self.datatype = datatype
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.vid2text, self.text_dict = {}, {}
        for item in load_jsonl(jsonl_path):
            self.vid2text.setdefault(item["vid_name"], []).append(item["desc_id"])
            self.text_dict[item["desc_id"]] = item["desc"]
        self.video_names = list(self.vid2text)
        self.rawVideoExtractor = RawVideoExtractorpyAV(
            size=image_resolution, num_segments=max_frames
        )

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        name = self.video_names[idx]
        video, _ = self.rawVideoExtractor.get_video_data(
            _video_path(self.video_dir, name, self.datatype)
        )
        text_names = self.vid2text[name]
        text_ids = [
            _tokenize(self.text_dict[n], self.tokenizer, self.max_words)
            for n in text_names
        ]
        return name, video, text_names, text_ids


def _pad_texts(texts):
    padded = torch.zeros(
        (len(texts), max(len(text) for text in texts)), dtype=torch.long
    )
    for idx, text in enumerate(texts):
        padded[idx, : len(text)] = text
    return padded


def collate_train(data):
    video_names, videos, text_names, text_ids = zip(*data)
    texts = [text for group in text_ids for text in group]
    labels = [idx for idx, group in enumerate(text_ids) for _ in group]
    return {
        "video_name": video_names,
        "video": torch.stack(videos),
        "text_name": [name for group in text_names for name in group],
        "text_ids": _pad_texts(texts),
        "text_labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_val_text(data):
    names, texts = zip(*data)
    return {"text_name": names, "text_ids": _pad_texts(texts)}


def get_train_dataloader(args, tokenizer):
    dataset = TrainDataset(
        args.prvr_train_annos,
        args.video_dir,
        tokenizer,
        max_words=args.max_words,
        max_frames=args.Nf,
        datatype=args.datatype,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_train,
    )
    return loader, len(dataset)


def get_val_vis_dataloader(args):
    dataset = Val_VideoDataset(
        args.prvr_val_annos, args.video_dir, max_frames=args.Nf, datatype=args.datatype
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return loader, len(dataset)


def get_val_txt_dataloader(args, tokenizer):
    dataset = Val_TextDataset(args.prvr_val_annos, tokenizer, max_words=args.max_words)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_val_text,
    )
    return loader, len(dataset)
