import numpy as np
from pyde265 import CodeStructure, Decoder, Image
import matplotlib.pyplot as plt
from skimage.draw import rectangle_perimeter


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


if __name__ == "__main__":

    decoder = Decoder()

    with open("examples/Kinkaku-ji.h265", 'rb') as file:
        for image in decoder.decode(file):
            image: Image
            code_structure = CodeStructure(image)

            reconstruction = image.get_image()
            h, w, _ = reconstruction.shape

            cb_info = ycbcr2rgb(reconstruction)

            for cb in code_structure.iter_code_blocks():
                rr, cc = rectangle_perimeter(start=cb.position + 1, extent=cb.size - 2)
                rr = np.clip(rr, a_min=0, a_max=h - 1)
                cc = np.clip(cc, a_min=0, a_max=w - 1)
                cb_info[rr, cc] = (255, 25, 25)

            pb_info = ycbcr2rgb(reconstruction)

            for pb in code_structure.iter_prediction_blocks():
                rr, cc = rectangle_perimeter(start=pb.position + 1, extent=pb.size - 2)
                rr = np.clip(rr, a_min=0, a_max=h - 1)
                cc = np.clip(cc, a_min=0, a_max=w - 1)
                pb_info[rr, cc] = (25, 255, 25)

            tb_info = ycbcr2rgb(reconstruction)

            for tb in code_structure.iter_transform_blocks():
                rr, cc = rectangle_perimeter(start=tb.position + 1, extent=tb.size - 2)
                rr = np.clip(rr, a_min=0, a_max=h - 1)
                cc = np.clip(cc, a_min=0, a_max=w - 1)
                tb_info[rr, cc] = (25, 25, 255)

            fig, ax = plt.subplots(2, 2, sharex="all", sharey="all")
            ax[0, 0].imshow(ycbcr2rgb(reconstruction))
            ax[0, 0].set_title("Reconstruction")
            ax[0, 0].axis('off')
            ax[0, 0].tick_params()
            ax[0, 1].imshow(cb_info)
            ax[0, 1].set_title("Code Blocks")
            ax[0, 1].axis('off')
            ax[1, 0].imshow(pb_info)
            ax[1, 0].set_title("Prediction Blocks")
            ax[1, 0].axis('off')
            ax[1, 1].imshow(tb_info)
            ax[1, 1].set_title("Transform Blocks")
            ax[1, 1].axis('off')
            plt.tight_layout()
            plt.show()
