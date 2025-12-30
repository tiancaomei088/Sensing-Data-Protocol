import pywt
import numpy as np


def wavelet_denoise_csi(csi_tensor):
    """
    :param: csi_tensor (np.ndarray): CSI data
    """
    # split amplitutde and phase
    amplitude = np.abs(csi_tensor)
    phase = np.angle(csi_tensor)
    
    denoised_amplitude = np.copy(amplitude)

    T, S, R = csi_tensor.shape

    def _denoise_channel(channel):
        try:
            # in case of dividing zero
            if np.std(channel) < 1e-6:
                return channel
            L = len(channel)

            w_name = 'db4'
            wavelet = pywt.Wavelet(w_name)
            max_level = pywt.dwt_max_level(L, wavelet.dec_len)
            
            if max_level < 1:
                w_name = 'db1'
                wavelet = pywt.Wavelet(w_name)
                max_level = pywt.dwt_max_level(L, wavelet.dec_len)
            
            if max_level < 1:
                return channel
            
            level = min(2, max_level)
            coeffs = pywt.wavedec(channel, wavelet, level=level)

            # calculate threshold of noise (VisuShrink)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(L))

            denoised_coeffs = [coeffs[0]] + [np.sign(c) * np.maximum(np.abs(c) - threshold, 0) for c in coeffs[1:]]
            
            # refactor
            denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
            
            return denoised_signal[:L]
        except Exception as e:
            print(f"wavalet denoising fail: {e}. original signal will be returned.")
            return channel

    for rx in range(R):
        for sc in range(S):
            denoised_amplitude[:, sc, rx] = _denoise_channel(amplitude[:, sc, rx])
            
    denoised_csi_tensor = denoised_amplitude * np.exp(1j * phase)
    
    return denoised_csi_tensor