import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from tqdm import tqdm

import numpy as np

import os

from tqdm import tqdm


# custom dataset used to load pairs for training
class Dataset(Dataset):
    def __init__(self, data_dir, pause_token, max_ema_len, decimation_factor):
        self.data_dir = data_dir
        self.max_ema_len = max_ema_len
        self.decimation_factor = decimation_factor

        label_files = os.listdir(os.path.join(data_dir, "lab"))

        self.ema_files = os.listdir(os.path.join(data_dir, "nema_npy"))
        self.ema_files.sort()

        print("Num samples: " + str(len(self.ema_files)))

        # loading phoneme sequence data + vocab info
        print("Loading phoneme data...")

        all_phone_list = []
        phone_sequences = []

        cleaned_names = []
        for file in label_files:
            if file[-3:] == "lab":
                cleaned_names += [file]

        cleaned_names.sort()

        for file in cleaned_names:
            with open(os.path.join(data_dir, "lab", file), "r") as f:
                sample_phone_seq = []
                for _, line in enumerate(f):
                    parsed_arr = line.split(" ")
                    if len(parsed_arr) > 1:
                        phone = parsed_arr[2].replace("\n", "")
                        sample_phone_seq += [phone]
                        all_phone_list += [phone]
                phone_sequences += [sample_phone_seq]

        self.phone_vocab = []

        for phone in all_phone_list:
            if phone not in self.phone_vocab:
                self.phone_vocab += [phone]

        print("Vocab length: " + str(len(self.phone_vocab)))

        # phone -> idx conversion
        self.phone_sequences = []
        self.seq_lengths = []

        for seq in phone_sequences:
            idx_seq = []
            for phone in seq:
                if (
                    len(idx_seq) > 0
                    and phone == pause_token
                    and idx_seq[-1] == self.phone_vocab.index(pause_token) + 1
                ):
                    pass
                else:
                    # add 1 because of 0 blank token index
                    idx_seq += [self.phone_vocab.index(phone) + 1]
            self.phone_sequences += [torch.tensor(idx_seq)]
            self.seq_lengths += [len(idx_seq)]

        # padding for batch training
        self.phone_sequences = pad_sequence(self.phone_sequences).permute(1, 0)

        print("Phone seq dims: " + str(self.phone_sequences.shape))

    def __len__(self):
        return len(self.ema_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        raw_data = torch.tensor(
            np.load(os.path.join(self.data_dir, "nema_npy", self.ema_files[idx])),
            dtype=torch.float32,
        )

        final_data = torch.zeros((self.max_ema_len, raw_data.shape[-1]))

        final_data[: raw_data.shape[0], :] = raw_data

        return (
            final_data,
            self.phone_sequences[idx].view(-1),
            torch.tensor(raw_data.shape[0] // self.decimation_factor),
            self.seq_lengths[idx],
        )
