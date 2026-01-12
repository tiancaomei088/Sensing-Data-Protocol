import os
import struct
import numpy as np

from sdp.readers.base import BaseReader
from sdp.structure import CSIData
from sdp.structure import BfeeFrame


class BfeeReader(BaseReader):
    def __init__(self):
        super().__init__()

    def read_file(self, file_path: str) -> CSIData:
        file_name = os.path.basename(file_path)
        ret_data = CSIData(file_name)

        with open(file_path, 'rb') as f:
            filesize = os.fstat(f.fileno()).st_size
            cur = 0
            while (cur + 3) < filesize:
                hdr = f.read(3)
                if len(hdr) < 3: break
                field_len = (hdr[0] << 8) | hdr[1]
                code = hdr[2]
                cur += 3
                if code == 0xBB:
                    payload = f.read(field_len - 1)
                    cur += (field_len - 1)
                    if len(payload) < (field_len - 1):
                        break
                    frame = self.parse_bfee_record(payload)
                    if frame is not None:
                        ret_data.add_frame(frame)
                else:
                    f.seek(field_len - 1, 1)
                    cur += (field_len - 1)
        print(f"[Info] {file_name}: B_FEE records={len(ret_data.frames)}")
        return ret_data

    def parse_bfee_record(self, payload: bytes):
        if len(payload) < 20:
            return None

        timestamp = (payload[0] |
                     (payload[1] << 8) |
                     (payload[2] << 16) |
                     (payload[3] << 24)) & 0xffffffff
        bfee_count = (payload[4] | (payload[5] << 8)) & 0xffff

        n_rx = payload[8]
        n_tx = payload[9]
        rssi_a = payload[10]
        rssi_b = payload[11]
        rssi_c = payload[12]
        noise = struct.unpack('b', payload[13:14])[0]
        agc = payload[14]
        antenna_sel = payload[15]
        csi_len = (payload[16] | (payload[17] << 8)) & 0xffff
        fake_rate = (payload[18] | (payload[19] << 8)) & 0xffff

        calc_len = (30 * (n_rx * n_tx * 8 * 2 + 3) + 7) // 8
        if csi_len != calc_len: return None
        if len(payload) < (20 + csi_len): return None

        csi_bytes = payload[20: 20 + csi_len]
        csi_array = np.zeros((30, n_rx, n_tx), dtype=np.complex64)

        bit_index = 0

        def get_bit(pos):
            byte_i = pos // 8
            if byte_i >= len(csi_bytes):
                return 0
            shift = pos % 8
            return (csi_bytes[byte_i] >> shift) & 0x1

        def get_bits_u8(pos):
            val = 0
            for b in range(8):
                val |= (get_bit(pos + b) << b)
            return val

        for sc_idx in range(30):
            bit_index += 3  # skip pilot
            for j in range(n_rx * n_tx):
                real8 = get_bits_u8(bit_index)
                imag8 = get_bits_u8(bit_index + 8)
                bit_index += 16
                if real8 & 0x80: real8 -= 256
                if imag8 & 0x80: imag8 -= 256
                rx_i = j % n_rx
                tx_i = j // n_rx
                csi_array[sc_idx, rx_i, tx_i] = np.complex64(real8 + 1j * imag8)
        return BfeeFrame(timestamp, csi_array, bfee_count, n_rx, n_tx, rssi_a, rssi_b, rssi_c,
                         noise, agc, antenna_sel, fake_rate)
