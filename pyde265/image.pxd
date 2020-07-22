from pyde265 cimport de265, de265_internals
cimport numpy as np

cdef class Image(object):

    cdef de265.de265_image* _image
    cdef de265.de265_decoder_context * _decoder_context
    cdef de265.de265_image_allocation * _image_allocation
    cdef list _available_signals
    cdef dict _y
    cdef dict _cb
    cdef dict _cr
    cdef int _ctb_unit_size
    cdef np.ndarray _ctb_info
    cdef int _cb_unit_size
    cdef np.ndarray _cb_info
    cdef int _pb_unit_size
    cdef np.ndarray _pb_info
    cdef int _intra_unit_size
    cdef np.ndarray _intra_info
    cdef int _tb_unit_size
    cdef np.ndarray _tb_info
    cdef int _prefetched

    @staticmethod
    cdef Image create(de265.de265_decoder_context *,  de265.de265_image *)

    cdef _prefetch(self)

