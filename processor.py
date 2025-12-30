import os
import re
import numpy as np

from typing import List
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from algorithms import phase_calibration, wavelet_denoise_csi
from structure.CSIData import CSIData


class Processor():
    def process(self, data_list: List[CSIData], **kwargs):
        dataset = kwargs.get('dataset', '')
        all_data = []
        all_labels = []
        all_groups = []
        worker_func = partial(_process_single_csi, dataset=dataset)
        with ProcessPoolExecutor(max_workers=32) as executor:
            results = executor.map(worker_func, data_list)
            for csi, label, group in results:
                if csi is not None:
                    all_data.append(csi)
                    all_labels.append(label)
                    all_groups.append(group)
        return all_data, all_labels, all_groups


# function for parallel processing
def _process_single_csi(csi_data, dataset):
    res = parse_file_info_from_filename(csi_data.file_name, dataset)
    label, group = selector(res, dataset)
    sorted_frames = sorted(csi_data.frames, key=lambda frame: frame.timestamp)
    frame_tensors = []
    for frame in sorted_frames:
        data = frame.csi_array
        frame_tensors.append(data)
    if frame_tensors:
        whole_csi = np.stack(frame_tensors, axis=0)
        whole_csi = whole_csi.squeeze()
        # discard data with too short time period(1 timestamp)
        if whole_csi.ndim < 3:
            print(f"only one timestamp: {csi_data.file_name} \n")
            return None, None, None
        else:
            whole_csi = phase_calibration(whole_csi)
            cleaned_csi = wavelet_denoise_csi(whole_csi)
            return cleaned_csi, label, group
    return None, None, None


def parse_file_info_from_filename(f_name, dataset):
    base = os.path.splitext(os.path.basename(f_name))[0]

    if dataset == 'widar':
        m = re.match(r'user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)', base)
        if m:
            user_id = int(m.group(1))
            gesture_type = int(m.group(2))
            torso_position = int(m.group(3))
            orientation = int(m.group(4))
            data_serial = int(m.group(5))
            receiver_number = int(m.group(6))
            return user_id, gesture_type, torso_position, orientation, data_serial, receiver_number
        else:
            print(f"[Warning] Skipping file {f_name}: Invalid format for Gesture Recognition.")

    elif dataset == 'gait':
        # Parse for Gait Recognition (pattern "user3-1-1-r1.dat")
        m = re.search(r'user(\d+)-(\d+)-(\d+)-r(\d+)', base, re.IGNORECASE)
        if m:
            user_id = int(m.group(1))
            track_id = int(m.group(2))
            data_serial = int(m.group(3))

            return user_id, track_id, data_serial, None, None, None
        else:
            print(f"[Warning] Skipping file {f_name}: Invalid format for Activity Recognition.")

    elif dataset == 'xrf55':
        m = re.search(r'(\d+)_(\d+)_', base)
        if m:
            user_id = int(m.group(1))
            action_id = int(m.group(2))
            return user_id, action_id, None, None, None, None
        else:
            print(f"[Warning] Skipping file {f_name}: Invalid format for xrf55.")

    elif dataset == 'elderAL':
        base = f_name.split('/')[4]
        m = re.search(r"user(\d+)_position(\d+)_activity(\d+)", base)
        if m:
            user_id = int(m.group(1))
            position_id = int(m.group(2))
            action_id = int(m.group(3))
            return user_id, position_id, action_id, None, None, None
        else:
            print(f"[Warning] Skipping file {f_name}: Invalid format for ElderAL Dataset.")

    else:
        print(f"[Error] Unknown task type: {dataset}")


def selector(res, dataset):
    label = None
    group = None

    if dataset == 'widar':
        label = int(res[1])
        group = int(res[2])
    elif dataset == 'gait':
        label = int(res[0])
        group = int(res[1])
    elif dataset == 'xrf55':
        label = int(res[1])
        group = int(res[0])
    elif dataset == 'elderAL':
        label = int(res[2])
        group = int(res[1])

    return label, group