from libc.stdint cimport (
    uint8_t, int8_t,
    uint16_t, int16_t,
    uint32_t, int32_t,
    uint64_t, int64_t
)

from pyde265 cimport de265, de265_internals

import numpy as np
from pyde265.de265_enums import ChromaFormat, InternalSignal
from typing import Union, Tuple


cdef class Image(object):

    def __cinit__(self):
        pass

    @staticmethod
    cdef Image create(de265.de265_image * image):
        cdef Image result = Image()
        result._image = image
        return result

    @property
    def height(self) -> int:
        return de265.de265_get_image_height(self._image, 0)

    @property
    def width(self) -> int:
        return de265.de265_get_image_width(self._image, 0)

    @property
    def chroma_format(self) -> ChromaFormat:
        return ChromaFormat(de265.de265_get_chroma_format(self._image))

    def y(self, signal: Union[InternalSignal, None] = None) -> np.ndarray:
        return self.get_plane(0, signal)

    def cb(self, signal: Union[InternalSignal, None] = None) -> np.ndarray:
        return self.get_plane(1, signal)

    def cr(self, signal: Union[InternalSignal, None] = None) -> np.ndarray:
        return self.get_plane(2, signal)

    # ToDo: This can only do 8 bit/px for now
    def get_plane(self, channel: int, signal: Union[InternalSignal, None] = None) -> np.ndarray:
        height = self.height
        width = self.width
        if channel in (1, 2):
            if self.chroma_format== ChromaFormat.C420:
                height //= 2
                width //= 2
            elif self.chroma_format == ChromaFormat.C422:
                width //= 2

        cdef int outstride = 0
        cdef uint8_t* buffer = <uint8_t *>0
        if signal is None:
            buffer = de265.de265_get_image_plane(self._image, channel, &outstride)
        else:
            buffer = de265_internals.de265_internals_get_image_plane(self._image, signal.value, channel, &outstride)

        plane = np.zeros(shape=(height, width), dtype=np.uint8)
        for h in range(height):
            plane[h] = np.frombuffer(buffer[h * outstride: h * outstride + width], count=width, dtype=np.uint8)
        return plane

    def get_image(self, signal: Union[InternalSignal, None] = None) -> np.ndarray:
        if self.chroma_format == ChromaFormat.C420:
            u = np.repeat(np.repeat(self.cb(signal), repeats=2, axis=1), repeats=2, axis=0)
            v = np.repeat(np.repeat(self.cr(signal), repeats=2, axis=1), repeats=2, axis=0)
        elif self.chroma_format == ChromaFormat.C422:
            u = np.repeat(self.cb(signal), repeats=2, axis=1)
            v = np.repeat(self.cr(signal), repeats=2, axis=1)
        else:
            u = self.cb(signal)
            v = self.cr(signal)
        return np.stack((self.y(signal), u, v), axis=-1)

    def get_ctb_slice_indices(self) -> Tuple[int, np.ndarray]:
        cdef int width, height, logsize
        de265_internals.de265_internals_get_CTB_Info_Layout(self._image, &width, &height, &logsize)
        ctb_size = 1 << logsize
        indices = np.ascontiguousarray(np.zeros(shape=(height, width), dtype=np.uint16))
        cdef uint16_t[:, :] memview = indices
        de265_internals.de265_internals_get_CTB_sliceIdx(self._image, &memview[0,0])
        return  ctb_size, indices

    def get_cb_info(self) -> Tuple[int, np.ndarray]:
        cdef int width, height, logsize
        de265_internals.de265_internals_get_CB_Info_Layout(self._image, &width, &height, &logsize)
        cb_size = 1 << logsize
        info = np.ascontiguousarray(np.zeros(shape=(height, width), dtype=np.uint16))
        cdef uint16_t[:, :] memview = info
        de265_internals.de265_internals_get_CB_info(self._image, &memview[0, 0])
        return cb_size, info

    def get_pb_info(self) -> Tuple[int, np.ndarray]:
        cdef int width, height, logsize
        de265_internals.de265_internals_get_PB_Info_layout(self._image, &width, &height, &logsize)
        pb_size = 1 << logsize
        info = np.ascontiguousarray(np.zeros(shape=(6, height, width), dtype=np.int16))
        cdef int16_t[:, :, :] memview = info
        # refPOC0, refPOC1, vec0_x, vec0_y, vec1_x, vec1_y
        de265_internals.de265_internals_get_PB_info(self._image, &memview[0, 0, 0], &memview[1, 0, 0],
                                                    &memview[2, 0, 0], &memview[3, 0, 0], &memview[4, 0, 0],
                                                    &memview[5, 0, 0])
        return pb_size, info

    def get_intra_info(self) -> Tuple[int, np.ndarray]:
        cdef int width, height, logsize
        de265_internals.de265_internals_get_IntraDir_Info_layout(self._image, &width, &height, &logsize)
        intra_size = 1 << logsize
        info = np.ascontiguousarray(np.zeros(shape=(2, height, width), dtype=np.uint8))
        cdef uint8_t[:, :, :] memview = info
        # Y intra direction, C intra direction
        de265_internals.de265_internals_get_intraDir_info(self._image, &memview[0, 0, 0], &memview[1, 0, 0])
        return intra_size, info

    def get_tu_info(self) -> Tuple[int, np.ndarray]:
        cdef int width, height, logsize
        de265_internals.de265_internals_get_TUInfo_Info_layout(self._image, &width, &height, &logsize)
        tu_size = 1 << logsize
        info = np.ascontiguousarray(np.zeros(shape=(height, width), dtype=np.uint8))
        cdef uint8_t[:, :] memview = info
        de265_internals.de265_internals_get_TUInfo_info(self._image, &memview[0, 0])
        return tu_size, info
