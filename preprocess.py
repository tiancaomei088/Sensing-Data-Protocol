import os
import readers
import numpy as np

from typing import List
from processor import Processor


def resize_csi_to_fixed_length(csi_samples_list: List[np.ndarray], target_length: int=1500, pad_value: float=0.0) -> List[np.ndarray]:
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


if __name__ == '__main__':
    input_path = "../data/elderAL"
    output_path = "../data/elderAL"
    dataset = 'elderAL'

    csi_data_list = readers.load_data(input_path, dataset)

    processor = Processor()
    res = processor.process(csi_data_list, dataset=dataset)

    unadjusted_data = res[0]
    processed_data = resize_csi_to_fixed_length(unadjusted_data, target_length=80)
    print(f"{processed_data[0].shape}")

    labels = res[1]
    groups = res[2]

    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    zero_indexed_labels = [label_map[label] for label in labels]

    unique_groups = sorted(list(set(groups)))
    group_map = {group: i for i, group in enumerate(unique_groups)}
    zero_indexed_groups = [group_map[group] for group in groups]

    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, 'processed_data.npz')
    np.savez_compressed(file_path, data=processed_data, labels=zero_indexed_labels, group=zero_indexed_groups)

    print(list(set(zero_indexed_labels)))
    print(list(set(zero_indexed_groups)))

    print(f"\n processed data is saved to: {file_path}")

