import numpy as np
import pyde265
from pyde265.de265_enums import InternalSignal
import matplotlib.pyplot as plt


def ycbcr2rgb(yuv_image: np.ndarray) -> np.ndarray:
    assert yuv_image.shape[-1] == 3, "YUV image does not have three channels in the last dimension"
    assert yuv_image.dtype == np.uint8, "YUV image doesn't have the correct data type"
    weights = np.zeros(shape=(3, 3), dtype=np.float32)
    weights[0] = (65.481 / 255.0, 128.553 / 255.0, 24.966 / 255.0)
    weights[1] = (-37.797 / 255.0, -74.203 / 255.0, 112.0 / 255.0)
    weights[2] = (112.0 / 255.0, -93.786 / 255.0, -18.214 / 255.0)
    weights = np.linalg.inv(weights.T)
    bias = np.array((16.0, 128.0, 128.0), dtype=np.float32)
    return np.clip(np.matmul(yuv_image.astype(np.float32) - bias, weights), 0, 255).astype(np.uint8)


decoder = pyde265.Decoder()
decoder.activate_internal_signal(signal=InternalSignal.PREDICTION)
decoder.activate_internal_signal(signal=InternalSignal.RESIDUAL)
decoder.activate_internal_signal(signal=InternalSignal.TR_COEFF)

with open("examples/Kinkaku-ji.h265", 'rb') as file:
    for image in decoder.decode(file):
        image: pyde265.Image
        reconstruction = image.get_image()
        prediction = image.get_image(signal=InternalSignal.PREDICTION)
        residual = image.get_image(signal=InternalSignal.RESIDUAL)
        tr_coeff = image.get_image(signal=InternalSignal.TR_COEFF).astype(np.float32)
        tr_coeff -= np.min(tr_coeff)
        tr_coeff /= np.max(tr_coeff)

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(ycbcr2rgb(reconstruction))
        ax[0, 0].set_title("Reconstruction")
        ax[0, 0].axis('off')
        ax[0, 0].tick_params()
        ax[0, 1].imshow(ycbcr2rgb(prediction))
        ax[0, 1].set_title("Prediction")
        ax[0, 1].axis('off')
        ax[1, 0].imshow(ycbcr2rgb(residual))
        ax[1, 0].set_title("Residual")
        ax[1, 0].axis('off')
        ax[1, 1].imshow(tr_coeff)
        ax[1, 1].set_title("Transform Coeff (scaled)")
        ax[1, 1].axis('off')
        plt.tight_layout()
        plt.show()
