from pyde265 cimport de265
from cpython cimport array
import array

cdef class Decoder(object):

    cdef de265.de265_decoder_context * _context

    cdef readonly int buffer_size
    cdef readonly int n_threads
    cdef readonly array.array signals_available
