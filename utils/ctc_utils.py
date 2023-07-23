import torch

import torchaudio

import numpy as np


def process_list(seq):
    list = []
    for i, char in enumerate(seq):
        if char != 0:
            if i != 0 and char == seq[i - 1]:  # remove duplicates
                pass
            else:
                list = list + [char]
    return list


def compute_error_rates(output, phone_sequences):
    estimated_sequences = torch.argmax(output, dim=-1)

    error_rates = []

    for phone_sequence, estimated_sequence in zip(phone_sequences, estimated_sequences):
        # proper per calculations need to remove padding
        phone_data_calc = phone_sequence[phone_sequence != 0].reshape((-1))

        per = torchaudio.functional.edit_distance(
            phone_data_calc[1:-1], process_list(estimated_sequence)[1:-1]
        ) / (len(phone_data_calc) - 2)

        error_rates += [per]

    return np.mean(np.array(error_rates))
