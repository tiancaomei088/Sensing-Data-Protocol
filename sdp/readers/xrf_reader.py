import numpy as np

from typing import List
from sdp.readers.base import BaseReader
from sdp.structure import CSIData
from sdp.structure import BaseFrame


class XrfReader(BaseReader):
    def __init__(self):
        super().__init__()

    def read_file(self, file_path) -> List[CSIData]:
        try:
            raw_data = np.load(file_path)
        except FileNotFoundError:
            print(f"cannot find file: {file_path}")
            return []

        try:
            reshaped_data = raw_data.reshape(3, 30, 3, 1000)
        except ValueError as e:
            print(f"reshape failed: {e}")
            return []

        csi_data_list = []
        num_receivers = 3
        num_time_steps = 1000

        for rx_idx in range(num_receivers):
            csi_data = CSIData(file_path)
            current_rx_data = reshaped_data[rx_idx]

            for timestamp in range(num_time_steps):
                csi_array = current_rx_data[:, :, timestamp]
                csi_array = csi_array.copy()
                frame = BaseFrame(timestamp=timestamp, csi_array=csi_array)
                
                csi_data.add_frame(frame)
            
            csi_data_list.append(csi_data)

        return csi_data_list