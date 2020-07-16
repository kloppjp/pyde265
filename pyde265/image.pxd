from pyde265 cimport de265, de265_internals


cdef class Image(object):

    cdef de265.de265_image* _image

    @staticmethod
    cdef Image create(de265.de265_image *)

