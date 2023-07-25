import torch
import torch.nn as nn

import torchaudio

from tqdm import tqdm

import numpy as np

import wandb

from utils.cnn_gru import Decoder
from utils.dataset import Dataset
from utils.ctc_utils import compute_error_rates


# config + hyperparams
DATA_DIR = "./data/cin_us_fjmw0/"
DATASET = "cin_us_fjmw0"
MODEL_NAME = "cnn_gru_larger_w_dropouts"

PAUSE_TOKEN = "pau"
DECIMATION_FACTOR = 2
MAX_EMA_LEN = int(1340 / DECIMATION_FACTOR)

device = torch.device("cuda:1")

EPOCHS = 100
lr = 1e-3
weight_decay = 1e-5

num_steps = 5


# model config
in_feature_dim = 12
conv_kernel = 2
conv_stride = 2
hidden_dim = 256
num_layers = 2
vocab_size = 43
dropout = 0.6


# wandb initialization
wandb.init(
    # set the wandb project where this run will be logged
    project="SSL-Gesture",
    # track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "architecture": MODEL_NAME,
        "dataset": DATASET,
        "epochs": EPOCHS,
    },
)


# decoder initialization
decoder = Decoder(
    in_feature_dim,
    conv_kernel,
    conv_stride,
    hidden_dim,
    num_layers,
    vocab_size,
    dropout,
)
decoder = decoder.to(device)


# dataloader initialization
# torch.manual_seed(0)

dataset = Dataset(DATA_DIR, PAUSE_TOKEN, MAX_EMA_LEN, DECIMATION_FACTOR)

train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])

batch_size = 32

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)


# optimizer + loss initialization
optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)

# ctc blank label is 0
criterion = nn.CTCLoss(zero_infinity=True)


# eval function
def eval_model():
    # decoder evaluation
    decoder.eval()

    train_losses = []
    train_error_rates = []
    for _, (
        ema_data,
        phone_sequences,
        ema_seq_lengths,
        target_seq_lengths,
    ) in enumerate(tqdm(train_loader)):
        ema_data = ema_data.to(device)
        phone_sequences = phone_sequences.to(device)
        ema_seq_lengths = ema_seq_lengths.to(device)
        target_seq_lengths = target_seq_lengths.to(device)

        output = decoder(ema_data)

        # for ctc loss
        output = output.permute(1, 0, 2)

        loss = criterion(
            output,
            phone_sequences,
            ema_seq_lengths,
            target_seq_lengths,
        )

        # greedy decoding per
        output = output.permute((1, 0, 2))

        error_rate = compute_error_rates(output, phone_sequences)
        train_error_rates += [error_rate]

        train_losses += [loss.item()]

        if _ == num_steps:
            loss = np.mean(np.array(train_losses))
            per = np.mean(np.array(train_error_rates))
            print("Train loss: " + str(loss))
            print("Train error: " + str(per))
            wandb.log(
                {
                    "train_loss_" + MODEL_NAME: loss,
                    "train_error_" + MODEL_NAME: per,
                }
            )

            break

    val_losses = []
    val_error_rates = []
    for ema_data, phone_sequences, ema_seq_lengths, target_seq_lengths in tqdm(
        val_loader
    ):
        ema_data = ema_data.to(device)
        phone_sequences = phone_sequences.to(device)
        ema_seq_lengths = ema_seq_lengths.to(device)
        target_seq_lengths = target_seq_lengths.to(device)

        output = decoder(ema_data)

        # for ctc loss
        output = output.permute(1, 0, 2)

        loss = criterion(
            output,
            phone_sequences,
            ema_seq_lengths,
            target_seq_lengths,
        )

        # greedy decoding per
        output = output.permute((1, 0, 2))

        error_rate = compute_error_rates(output, phone_sequences)
        val_error_rates += [error_rate]

        val_losses += [loss.item()]
    loss = np.mean(np.array(val_losses))
    per = np.mean(np.array(val_error_rates))
    print("Val loss: " + str(loss))
    print("Val error: " + str(per))
    wandb.log(
        {
            "val_loss_" + MODEL_NAME: loss,
            "val_error_" + MODEL_NAME: per,
        }
    )


# training loop
# eval_model()  # initial eval

for epoch in range(EPOCHS):
    print("EPOCH: " + str(epoch))

    # decoder train
    decoder.train()

    for ema_data, phone_sequences, ema_seq_lengths, target_seq_lengths in tqdm(
        train_loader
    ):
        ema_data = ema_data.to(device)
        phone_sequences = phone_sequences.to(device)
        ema_seq_lengths = ema_seq_lengths.to(device)
        target_seq_lengths = target_seq_lengths.to(device)

        output = decoder(ema_data)

        # for ctc loss
        output = output.permute(1, 0, 2)

        loss = criterion(
            output,
            phone_sequences,
            ema_seq_lengths,
            target_seq_lengths,
        )

        optimizer.zero_grad()
        nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)
        loss.backward()
        optimizer.step()

    eval_model()

    if epoch % 50 == 0 and epoch != 0:
        torch.save(decoder, "./saved_models/" + MODEL_NAME + "_decoder")

torch.save(decoder, "./saved_models/" + MODEL_NAME + "_decoder")
