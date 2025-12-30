import numpy as np

from dataclasses import dataclass, field


@dataclass
class BaseFrame:
    """
    Base class for CSI Frames
    """
    timestamp: str
    csi_array: np.ndarray = field(repr=False)

    def __repr__(self):
        return (f"timestamp={self.timestamp}, "
                f"csi_shape={self.csi_array.shape}, "
                f"dtype={self.csi_array.dtype})")




class BfeeFrame(BaseFrame):
    """
    Represents a Widar Bfee frame.
    """
    def __init__(self, timestamp, csi_array, bfee_count, n_rx, n_tx,
                 rssi_a, rssi_b, rssi_c, noise, agc, antenna_sel, fake_rate):
        super().__init__(timestamp=timestamp, csi_array=csi_array)
        self.bfee_count = bfee_count
        self.n_rx = n_rx
        self.n_tx = n_tx
        self.rssi_a = rssi_a
        self.rssi_b = rssi_b
        self.rssi_c = rssi_c
        self.noise = noise
        self.agc = agc
        self.antenna_sel = antenna_sel
        self.fake_rate = fake_rate