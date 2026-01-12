import re
import csv
import numpy as np

from sdp.readers.base import BaseReader
from sdp.structure import CSIData
from sdp.structure import BaseFrame


class ElderReader(BaseReader):
    def __init__(self):
        super().__init__()

    def read_file(self, file_path: str) -> CSIData:
        csi_data = CSIData(file_name=file_path)
        pattern = re.compile(r"amp_tx(\d+)_rx(\d+)_sub(\d+)")
        target_tx = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            try:
                headers = next(reader)
            except StopIteration:
                print("empty file")
                return []

            # key=CSV column, value=(sub_idx, rx_idx)
            col_mapping = {} 
            timestamp_idx = -1
            
            max_sub = -1
            max_rx = -1
            
            for idx, col_name in enumerate(headers):
                col_name = col_name.strip()
                if col_name == 'timestamp':
                    timestamp_idx = idx
                    continue
                
                match = pattern.match(col_name)
                if match:
                    tx = int(match.group(1))
                    rx = int(match.group(2))
                    sub = int(match.group(3))
                    
                    if tx == target_tx:
                        col_mapping[idx] = (sub, rx)
                        if sub > max_sub: max_sub = sub
                        if rx > max_rx: max_rx = rx
            
            if timestamp_idx == -1:
                raise ValueError("cannot fime column 'timestamp'")
                
            num_sub = max_sub + 1
            num_rx = max_rx + 1
            
            row_count = 0
            for row in reader:
                if not row: continue
                
                try:
                    ts_str = row[timestamp_idx]
                    timestamp = float(ts_str) if '.' in ts_str else int(ts_str)
                    
                    csi_array = np.zeros((num_sub, num_rx))
                    
                    for col_idx, (sub, rx) in col_mapping.items():
                        val = float(row[col_idx])
                        csi_array[sub, rx] = val
                    
                    frame = BaseFrame(timestamp=timestamp, csi_array=csi_array)
                    csi_data.add_frame(frame)
                    
                    row_count += 1
                    
                except ValueError as e:
                    print(f"parse error at row {row_count+2}: {e}")
                    continue

        return csi_data