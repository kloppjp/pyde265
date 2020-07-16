from pyde265.image import Image
import numpy as np
from typing import Iterable, Tuple


# Returns Height, Width, H Offset, W Offset
def _pb_position_size(mode: int, cb_size_px: int) -> Iterable[Tuple[int, int, int, int]]:
    if mode == 0:  # 2N x 2N
        yield cb_size_px, cb_size_px, 0, 0
    elif mode == 1:  # N x 2N
        yield cb_size_px // 2, cb_size_px, 0, 0
        yield cb_size_px // 2, cb_size_px, cb_size_px // 2, 0
    elif mode == 2:  # 2N x N
        yield cb_size_px, cb_size_px // 2, 0, 0
        yield cb_size_px, cb_size_px // 2, 0, cb_size_px // 2
    elif mode == 3:  # N x N
        yield cb_size_px // 2, cb_size_px // 2, 0, 0
        yield cb_size_px // 2, cb_size_px // 2, cb_size_px // 2, 0
        yield cb_size_px // 2, cb_size_px // 2, cb_size_px // 2, cb_size_px // 2
        yield cb_size_px // 2, cb_size_px // 2, 0, cb_size_px // 2
    elif mode == 4:  # nU x 2N
        yield cb_size_px // 4, cb_size_px, 0, 0
        yield cb_size_px // 4 * 3, cb_size_px, cb_size_px // 4, 0
    elif mode == 5:  # nD x 2N
        yield cb_size_px // 4 * 3, cb_size_px, 0, 0
        yield cb_size_px // 4, cb_size_px, cb_size_px // 4 * 3, 0
    elif mode == 6:  # 2N x nL
        yield cb_size_px, cb_size_px // 4, 0, 0
        yield cb_size_px, cb_size_px // 4 * 3, 0, cb_size_px // 4
    elif mode == 7:  # 2N x nR
        yield cb_size_px, cb_size_px // 4 * 3, 0, 0
        yield cb_size_px, cb_size_px // 4, 0, cb_size_px // 4 * 3


_intra_directions = np.array((
    (0, 0), (0, 0), (32, -32), (32, -26), (32, -21), (32, -17), (32, -13), (32, -9), (32, -5), (32, -2), (32, 0),
    (32, 2), (32, 5), (32, 9), (32, 13), (32, 17), (32, 21), (32, 26), (32, 32),
    (26, 32), (21, 32), (17, 32), (13, 32), (9, 32), (5, 32), (2, 32), (0, 32),
    (-2, 32), (-5, 32), (-9, 32), (-13, 32), (-17, 32), (-21, 32), (-26, 32), (-32, 32)))


# This is based on https://github.com/IENT/YUView/blob/master/YUViewLib/src/decoder/decoderLibde265.cpp
class CodeStructure:

    def __init__(self, image: Image):
        assert image is not None, "CodeStructure must be initialised from a non-None image"
        self._image = image
        self._parse()

    def _parse(self):
        # Code Blocks
        cb_unit_size, cb_info = self._image.get_cb_info()
        self.cb_unit_size = cb_unit_size

        log_size = np.bitwise_and(cb_info, 7)  # Bits 1, 2, 3
        self.cb_available = log_size > 0
        self.cb_size = np.ones_like(log_size) << log_size
        self.cb_pb_partitioning = np.bitwise_and(cb_info, 8 + 16 + 32)  # Bits 4, 5, 6
        self.cb_prediction_mode = np.bitwise_and(cb_info, 64 + 128)  # Bits 7, 8
        self.cb_pcm_flag = np.bitwise_and(cb_info, 256)  # Bit 9
        self.cb_tqbypass_flag = np.bitwise_and(cb_info, 512)  # Bit 10

        # Copy values to other code blocks so that in all arrays except cb_available any two elements belonging to the
        # same code block do have the same values
        for idx, val in np.ndenumerate(self.cb_available):
            if not val:
                continue
            n_units = self.cb_size[idx] // self.cb_unit_size
            if n_units > 1:
                for array in (self.cb_size, self.cb_pb_partitioning, self.cb_prediction_mode, self.cb_pcm_flag,
                              self.cb_tqbypass_flag):
                    array[idx[0]:idx[0] + n_units, idx[1]:idx[1] + n_units] = array[idx]

        # Prediction Blocks
        pb_unit_size, pb_info = self._image.get_pb_info()
        self.pb_unit_size = pb_unit_size
        self.pb_poc0_idx = pb_info[np.newaxis, 0]
        self.pb_poc1_idx = pb_info[np.newaxis, 1]
        self.pb_vec0 = pb_info[2:4]
        self.pb_vec1 = pb_info[4:]
        self.pb_size = np.zeros(shape=(2, pb_info.shape[1], pb_info.shape[2]), dtype=np.int)
        self.pb_available = np.zeros(shape=pb_info.shape[1:], dtype=np.int)

        # Copy values to other code blocks so that in all arrays except for pb_available any two elements belonging to
        # the same prediction block do have the same values
        for idx, val in np.ndenumerate(self.cb_available):
            if not val:
                continue
            if self.cb_prediction_mode[idx] == 0:
                continue
            for h, w, h_off, w_off in _pb_position_size(self.cb_pb_partitioning[idx], self.cb_size[idx]):
                h_units = h // self.pb_unit_size
                w_units = w // self.pb_unit_size
                x_units = (idx[0] * self.cb_unit_size + h_off) // self.pb_unit_size
                y_units = (idx[1] * self.cb_unit_size + w_off) // self.pb_unit_size
                for array in (self.pb_poc0_idx, self.pb_poc1_idx, self.pb_vec0, self.pb_vec1):
                    array[:, x_units:x_units + h_units, y_units:y_units + w_units] = array[:, x_units, y_units]
                array[:, x_units:x_units + h_units, y_units:y_units + w_units] = (h, w)
                array[x_units, y_units] = 1

        # Transform Blocks
        tb_unit_size, tb_info = self._image.get_tu_info()
        self.tb_unit_size = tb_unit_size
        intra_unit_size, intra_info = self._image.get_intra_info()
        self.intra_unit_size = intra_unit_size

        self.tb_available = np.zeros(shape=tb_info.shape, dtype=np.int)
        self.tb_depth = np.ones(shape=tb_info.shape, dtype=np.int)
        self.tb_is_intra = np.zeros(shape=tb_info.shape, dtype=np.int)
        self.tb_intra_y_dir = np.zeros(shape=(2, tb_info.shape[0], tb_info.shape[1]), dtype=np.int)
        self.tb_intra_c_dir = np.zeros(shape=(2, tb_info.shape[0], tb_info.shape[1]), dtype=np.int)

        for idx, val in np.ndenumerate(self.cb_prediction_mode):
            if val == 0:
                self.tb_is_intra[idx[0] * cb_unit_size // tb_unit_size, idx[1] * cb_unit_size // tb_unit_size] = 1

        tb_field_h, tb_field_w = tb_info.shape
        for idx, val in np.ndenumerate(self.tb_depth):
            for lvl in 1, 2, 3:
                depth = 2 ** lvl
                block_size = cb_unit_size // tb_unit_size // depth
                if tb_info[idx] & (depth // 2):
                    for array, value in zip((self.tb_depth, self.tb_is_intra), (depth, self.tb_is_intra[idx])):
                        array[idx] = value
                        if idx[0] + block_size < tb_field_h:
                            array[idx[0] + block_size, idx[1]] = value
                            if idx[1] + block_size < tb_field_w:
                                array[idx[0] + block_size, idx[1] + block_size] = value
                        if idx[1] + block_size < tb_field_w:
                            array[idx[0], idx[1] + block_size] = value
                else:
                    self.tb_available[idx] = 1
            if self.tb_available[idx] == 1:
                intra_idx = (idx[0] * tb_unit_size // intra_unit_size, idx[1] * tb_unit_size // intra_unit_size)
                block_size = cb_unit_size // tb_unit_size // self.tb_depth[idx] // 2  # Block size in tb units
                block_size_px = block_size * tb_unit_size
                luma_vector_idx = intra_info[0][intra_idx]
                if luma_vector_idx <= 34:
                    self.tb_intra_y_dir[:, idx[0]:idx[0] + block_size, idx[1]:idx[1] + block_size] = _intra_directions[
                        luma_vector_idx][:, np.newaxis, np.newaxis] * block_size_px / 4
                chroma_vector_idx = intra_info[1][intra_idx]
                if chroma_vector_idx <= 34:
                    self.tb_intra_c_dir[:, idx[0]:idx[0] + block_size, idx[1]:idx[1] + block_size] = _intra_directions[
                        chroma_vector_idx][:, np.newaxis, np.newaxis] * block_size_px / 4

