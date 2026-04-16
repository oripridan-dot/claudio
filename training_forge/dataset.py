import os

import torch
from torch.utils.data import DataLoader, Dataset


class IntentAudioDataset(Dataset):
    """
    Dataset loader for Claudio Intent features extracted via `extract_dataset.py`.
    Expects .pt files containing dictionary: {f0, loudness, mfcc, audio}
    """
    def __init__(self, data_dir, clip_frames=250):
        self.data_dir = data_dir
        self.clip_frames = clip_frames
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu')

        # Truncate or pad to exactly `clip_frames` length
        # At 250Hz frame rate, 250 frames = 1.0 second context
        # Audio length will be 250 * 192 (hop) = 48000

        hop = 192
        f0 = data['f0']
        loudness = data['loudness']
        z = data['z']
        audio = data['audio']

        actual_frames = f0.shape[0]

        if actual_frames > self.clip_frames:
            # Crop random window
            start = torch.randint(0, actual_frames - self.clip_frames, (1,)).item()
            f0 = f0[start:start+self.clip_frames]
            loudness = loudness[start:start+self.clip_frames]
            z = z[start:start+self.clip_frames]
            a_start = start * hop
            audio = audio[a_start:a_start+(self.clip_frames*hop)]
        elif actual_frames < self.clip_frames:
            # Pad zeroes
            pad_frames = self.clip_frames - actual_frames
            f0 = torch.nn.functional.pad(f0, (0, 0, 0, pad_frames))
            loudness = torch.nn.functional.pad(loudness, (0, 0, 0, pad_frames))
            z = torch.nn.functional.pad(z, (0, 0, 0, pad_frames))
            pad_audio = pad_frames * hop
            audio = torch.nn.functional.pad(audio, (0, pad_audio))

        return {
            'f0': f0,
            'loudness': loudness,
            'z': z,
            'audio': audio
        }

def get_dataloader(data_dir, batch_size=8, shuffle=True):
    dataset = IntentAudioDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
