#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import torch
import torchaudio
import torchvision


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
        vision_encoder=None,
        audio_encoder=None,
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio
        
        # Support for new encoder-based modality detection
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        
        # Determine actual modality based on encoders if provided
        if vision_encoder and audio_encoder:
            self.effective_modality = "multimodal"
        elif vision_encoder:
            self.effective_modality = "video"
        elif audio_encoder:
            self.effective_modality = "audio"
        else:
            # Fall back to legacy modality argument
            self.effective_modality = modality

        self.list = self.load_list(label_path)
        self.input_lengths = [int(_[2]) for _ in self.list]

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            paths_counts_labels.append((dataset_name, rel_path, int(input_length), torch.tensor([int(_) for _ in token_id.split()])))
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        
        if self.effective_modality == "video":
            video = load_video(path)
            video = self.video_transform(video)
            return {"input": video, "target": token_id}
        elif self.effective_modality == "audio":
            audio = load_audio(path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}
        elif self.effective_modality == "multimodal":
            # Load both video and audio for multimodal processing
            video = load_video(path)
            video = self.video_transform(video)
            audio = load_audio(path)
            audio = self.audio_transform(audio)
            return {"input": video, "audio_input": audio, "target": token_id}
        else:
            # Fallback to legacy behavior for backward compatibility
            if self.modality == "video":
                video = load_video(path)
                video = self.video_transform(video)
                return {"input": video, "target": token_id}
            elif self.modality == "audio":
                audio = load_audio(path)
                audio = self.audio_transform(audio)
                return {"input": audio, "target": token_id}

    def __len__(self):
        return len(self.list)
