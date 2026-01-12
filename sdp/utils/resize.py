import numpy as np
from typing import List

def resize_csi_to_fixed_length(csi_samples_list: List[np.ndarray], target_length: int = 1500, pad_value: float = 0.0) -> \
List[np.ndarray]:
    if not csi_samples_list:
        return []

    resized_samples_list = []
    for sample in csi_samples_list:
        current_T = sample.shape[0]

        if current_T > target_length:
            resized_sample = sample[:target_length, :, :]
        elif current_T < target_length:
            pad_width = target_length - current_T
            padding_config = ((0, pad_width), (0, 0), (0, 0))
            resized_sample = np.pad(sample,
                                    pad_width=padding_config,
                                    mode='constant',
                                    constant_values=pad_value)
        else:
            resized_sample = sample

        resized_samples_list.append(resized_sample)

    return resized_samples_list