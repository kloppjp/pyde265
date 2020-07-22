from libc.stdint cimport (
    uint8_t, int8_t,
    uint16_t, int16_t,
    uint32_t, int32_t,
    uint64_t, int64_t
)

from pyde265 cimport de265, de265_internals

import numpy as np
from pyde265.de265_enums import ChromaFormat, InternalSignal
from typing import Union, Tuple, List

cdef class Image(object):

    def __cinit__(self):
        self._image_allocation = de265.de265_get_default_image_allocation_functions()

    def __init__(self):
        self._available_signals = None
        self._y = dict()
        self._cb = dict()
        self._cr = dict()
        self._cb_unit_size = 0
        self._ctb_unit_size = 0
        self._pb_unit_size = 0
        self._tb_unit_size = 0
        self._intra_unit_size = 0
        self._ctb_info = None
        self._cb_info = None
        self._intra_info = None
        self._pb_info = None
        self._tb_info = None
        self._prefetched = False

    def __del__(self):
        for signal in [None] + self.available_signals:
            del self._y[signal]
            del self._cb[signal]
            del self._cr[signal]
        for o in (self._ctb_info, self._cb_info, self._intra_info, self._pb_info, self._tb_info):
            if o is not None:
                del o

    def __dealloc__(self):
        cdef int user_data = 0
        self._image_allocation.release_buffer(self._decoder_context, self._image, &user_data)

    @staticmethod
    cdef Image create(de265.de265_decoder_context * decoder_context, de265.de265_image * image):
        cdef Image result = Image()
        result._image = image
        result._decoder_context = decoder_context
        return result

    @property
    def height(self) -> int:
        return de265.de265_get_image_height(self._image, 0)

    @property
    def width(self) -> int:
        return de265.de265_get_image_width(self._image, 0)

    @property
    def available_signals(self) -> List[InternalSignal]:
        return self._available_signals

    @available_signals.setter
    def available_signals(self, available_signals: List[InternalSignal]):
        if self._available_signals is None:
            self._available_signals = available_signals

    @property
    def chroma_format(self) -> ChromaFormat:
        return ChromaFormat(de265.de265_get_chroma_format(self._image))

    def y(self, signal: Union[InternalSignal, None] = None) -> np.ndarray:
        if signal not in self._y.keys():
            self._y[signal] = self._fetch_plane(0, signal)
        return self._y[signal]

    def cb(self, signal: Union[InternalSignal, None] = None) -> np.ndarray:
        if signal not in self._cb.keys():
            self._cb[signal] = self._fetch_plane(1, signal)
        return self._cb[signal]

    def cr(self, signal: Union[InternalSignal, None] = None) -> np.ndarray:
        if signal not in self._cr.keys():
            self._cr[signal] = self._fetch_plane(2, signal)
        return self._cr[signal]

    cdef _prefetch(self):
        if self._prefetched:
            return
        for signal in [None] + self.available_signals:
            self.y(signal)
            self.cb(signal)
            self.cr(signal)
        self.get_ctb_slice_indices()
        self.get_cb_info()
        self.get_pb_info()
        self.get_intra_info()
        self.get_tb_info()
        self._prefetched = True

    # ToDo: This can only do 8 bit/px for now
    def _fetch_plane(self, channel: int, signal: Union[InternalSignal, None] = None) -> np.ndarray:
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
        cdef uint16_t[:, :] memview
        if self._ctb_unit_size == 0:
            de265_internals.de265_internals_get_CTB_Info_Layout(self._image, &width, &height, &logsize)
            ctb_size = 1 << logsize
            indices = np.ascontiguousarray(np.zeros(shape=(height, width), dtype=np.uint16))
            memview = indices
            de265_internals.de265_internals_get_CTB_sliceIdx(self._image, &memview[0,0])
            self._ctb_unit_size = ctb_size
            self._ctb_info = indices
        return  self._ctb_unit_size, self._ctb_unit_size

    def get_cb_info(self) -> Tuple[int, np.ndarray]:
        cdef int width, height, logsize
        cdef uint16_t[:, :] memview
        if self._cb_unit_size == 0:
            de265_internals.de265_internals_get_CB_Info_Layout(self._image, &width, &height, &logsize)
            cb_size = 1 << logsize
            info = np.ascontiguousarray(np.zeros(shape=(height, width), dtype=np.uint16))
            memview = info
            de265_internals.de265_internals_get_CB_info(self._image, &memview[0, 0])
            self._cb_unit_size = cb_size
            self._cb_info = info
        return self._cb_unit_size, self._cb_info

    def get_pb_info(self) -> Tuple[int, np.ndarray]:
        cdef int width, height, logsize
        cdef int16_t[:, :, :] memview
        if self._pb_unit_size == 0:
            de265_internals.de265_internals_get_PB_Info_layout(self._image, &width, &height, &logsize)
            pb_size = 1 << logsize
            info = np.ascontiguousarray(np.zeros(shape=(6, height, width), dtype=np.int16))
            memview = info
            # refPOC0, refPOC1, vec0_x, vec0_y, vec1_x, vec1_y
            de265_internals.de265_internals_get_PB_info(self._image, &memview[0, 0, 0], &memview[1, 0, 0],
                                                        &memview[2, 0, 0], &memview[3, 0, 0], &memview[4, 0, 0],
                                                        &memview[5, 0, 0])
            self._pb_unit_size = pb_size
            self._pb_info = info
        return self._pb_unit_size, self._pb_info

    def get_intra_info(self) -> Tuple[int, np.ndarray]:
        cdef int width, height, logsize
        cdef uint8_t[:, :, :] memview
        if self._intra_unit_size == 0:
            de265_internals.de265_internals_get_IntraDir_Info_layout(self._image, &width, &height, &logsize)
            intra_size = 1 << logsize
            info = np.ascontiguousarray(np.zeros(shape=(2, height, width), dtype=np.uint8))
            memview = info
            # Y intra direction, C intra direction
            de265_internals.de265_internals_get_intraDir_info(self._image, &memview[0, 0, 0], &memview[1, 0, 0])
            self._intra_unit_size = intra_size
            self._intra_info = info
        return self._intra_unit_size, self._intra_info

    def get_tb_info(self) -> Tuple[int, np.ndarray]:
        cdef int width, height, logsize
        cdef uint8_t[:, :] memview
        if self._tb_unit_size == 0:
            de265_internals.de265_internals_get_TUInfo_Info_layout(self._image, &width, &height, &logsize)
            tu_size = 1 << logsize
            info = np.ascontiguousarray(np.zeros(shape=(height, width), dtype=np.uint8))
            memview = info
            de265_internals.de265_internals_get_TUInfo_info(self._image, &memview[0, 0])
            self._tb_unit_size = tu_size
            self._tb_info = info
        return self._tb_unit_size, self._tb_info
