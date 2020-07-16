from libc.stdint cimport (
    uint8_t, int8_t,
    uint16_t, int16_t,
    uint32_t, int32_t,
    uint64_t, int64_t
)
from pyde265 cimport de265

cdef extern from "de265_internals.h":

    cdef enum de265_internals_param:
        DE265_INTERNALS_DECODER_PARAM_SAVE_PREDICTION=0,
        DE265_INTERNALS_DECODER_PARAM_SAVE_RESIDUAL=1,
        DE265_INTERNALS_DECODER_PARAM_SAVE_TR_COEFF=2

    cdef void de265_internals_set_parameter_bool(de265.de265_decoder_context* context, de265_internals_param param, int value)
    cdef uint8_t* de265_internals_get_image_plane(de265.de265_image* img, de265_internals_param signal, int channel,
                                                  int* out_stride)

    cdef void de265_internals_get_CTB_Info_Layout(de265.de265_image *img, int *widthInUnits, int *heightInUnits, int *log2UnitSize)

    cdef void de265_internals_get_CTB_sliceIdx(de265.de265_image *img, uint16_t *idxArray)

    cdef void de265_internals_get_CB_Info_Layout(de265.de265_image *img, int *widthInUnits, int *heightInUnits, int *log2UnitSize)

# Get the coding block info. The values are stored in the following way:
# This function returns an array of values, sorted in a grid.
# You can get the layout of the grid by calling internals_get_CB_Info_Layout.
# The values are then arranged in a raster scan so the conversion from
# a unit position (x,y) to an index is: y*widthInUnits+x
#
# Each value in this array contains the following infomation:
# Bits 0:2 - The cb size in log2 pixels. Every CB can span multiple values in this array.
#                 Only the top left most unit contains a value. All others are set to 0. (3 Bits)
# Bits 3:5 - The part size (3 Bits)
# Bits 6:7 - The prediction mode (0: INTRA, 1: INTER, 2: SKIP) (2 Bits)
# Bit  8   - PCM flag (1 Bit)
# Bit  9   - CU Transquant bypass flag (1 Bit)

    cdef void de265_internals_get_CB_info(de265.de265_image *img, uint16_t *idxArray)

    cdef void de265_internals_get_PB_Info_layout(de265.de265_image *img, int *widthInUnits, int *heightInUnits, int *log2UnitSize)

    cdef void de265_internals_get_PB_info(de265.de265_image *img, int16_t *refPOC0, int16_t *refPOC1, int16_t *x0,
                                         int16_t *y0, int16_t *x1, int16_t *y1)

    cdef void de265_internals_get_IntraDir_Info_layout(de265.de265_image *img, int *widthInUnits, int *heightInUnits,
                                                       int *log2UnitSize)

    cdef void de265_internals_get_intraDir_info(de265.de265_image *img, uint8_t *intraDir, uint8_t *intraDirChroma)

    cdef void de265_internals_get_TUInfo_Info_layout(de265.de265_image *img, int *widthInUnits, int *heightInUnits,
                                                     int *log2UnitSize)

    cdef void de265_internals_get_TUInfo_info(de265.de265_image *img, uint8_t *tuInfo)
