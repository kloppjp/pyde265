from libc.stdint cimport (
    uint8_t, int8_t,
    uint16_t, int16_t,
    uint32_t, int32_t,
    uint64_t, int64_t
)

cdef extern from "de265.h":

    int de265_get_version_number()

    ctypedef enum de265_error:
        DE265_OK = 0,
        DE265_ERROR_NO_SUCH_FILE=1,
        DE265_ERROR_COEFFICIENT_OUT_OF_IMAGE_BOUNDS=4,
        DE265_ERROR_CHECKSUM_MISMATCH=5,
        DE265_ERROR_CTB_OUTSIDE_IMAGE_AREA=6,
        DE265_ERROR_OUT_OF_MEMORY=7,
        DE265_ERROR_CODED_PARAMETER_OUT_OF_RANGE=8,
        DE265_ERROR_IMAGE_BUFFER_FULL=9,
        DE265_ERROR_CANNOT_START_THREADPOOL=10,
        DE265_ERROR_LIBRARY_INITIALIZATION_FAILED=11,
        DE265_ERROR_LIBRARY_NOT_INITIALIZED=12,
        DE265_ERROR_WAITING_FOR_INPUT_DATA=13,
        DE265_ERROR_CANNOT_PROCESS_SEI=14,
        DE265_ERROR_PARAMETER_PARSING=15,
        DE265_ERROR_NO_INITIAL_SLICE_HEADER=16,
        DE265_ERROR_PREMATURE_END_OF_SLICE=17,
        DE265_ERROR_UNSPECIFIED_DECODING_ERROR=18,
        DE265_ERROR_NOT_IMPLEMENTED_YET = 502,
        DE265_WARNING_NO_WPP_CANNOT_USE_MULTITHREADING = 1000,
        DE265_WARNING_WARNING_BUFFER_FULL=1001,
        DE265_WARNING_PREMATURE_END_OF_SLICE_SEGMENT=1002,
        DE265_WARNING_INCORRECT_ENTRY_POINT_OFFSET=1003,
        DE265_WARNING_CTB_OUTSIDE_IMAGE_AREA=1004,
        DE265_WARNING_SPS_HEADER_INVALID=1005,
        DE265_WARNING_PPS_HEADER_INVALID=1006,
        DE265_WARNING_SLICEHEADER_INVALID=1007,
        DE265_WARNING_INCORRECT_MOTION_VECTOR_SCALING=1008,
        DE265_WARNING_NONEXISTING_PPS_REFERENCED=1009,
        DE265_WARNING_NONEXISTING_SPS_REFERENCED=1010,
        DE265_WARNING_BOTH_PREDFLAGS_ZERO=1011,
        DE265_WARNING_NONEXISTING_REFERENCE_PICTURE_ACCESSED=1012,
        DE265_WARNING_NUMMVP_NOT_EQUAL_TO_NUMMVQ=1013,
        DE265_WARNING_NUMBER_OF_SHORT_TERM_REF_PIC_SETS_OUT_OF_RANGE=1014,
        DE265_WARNING_SHORT_TERM_REF_PIC_SET_OUT_OF_RANGE=1015,
        DE265_WARNING_FAULTY_REFERENCE_PICTURE_LIST=1016,
        DE265_WARNING_EOSS_BIT_NOT_SET=1017,
        DE265_WARNING_MAX_NUM_REF_PICS_EXCEEDED=1018,
        DE265_WARNING_INVALID_CHROMA_FORMAT=1019,
        DE265_WARNING_SLICE_SEGMENT_ADDRESS_INVALID=1020,
        DE265_WARNING_DEPENDENT_SLICE_WITH_ADDRESS_ZERO=1021,
        DE265_WARNING_NUMBER_OF_THREADS_LIMITED_TO_MAXIMUM=1022,
        DE265_NON_EXISTING_LT_REFERENCE_CANDIDATE_IN_SLICE_HEADER=1023,
        DE265_WARNING_CANNOT_APPLY_SAO_OUT_OF_MEMORY=1024,
        DE265_WARNING_SPS_MISSING_CANNOT_DECODE_SEI=1025,
        DE265_WARNING_COLLOCATED_MOTION_VECTOR_OUTSIDE_IMAGE_AREA=1026,
        DE265_WARNING_MULTILAYER_ERROR_SWITCH_TO_BASE_LAYER=1027,
        DE265_WARNING_MULTILAYER_NON_ZERO_MV_FOR_INTER_LAYER_PREDICTION=1028

    cdef char * de265_get_error_text(de265_error)
    cdef int de265_isOK(de265_error)
    cdef void de265_set_verbosity(int)

    cdef struct de265_image:
        pass

    ctypedef void de265_decoder_context;

    cdef enum de265_chroma:
        de265_chroma_mono=0,
        de265_chroma_420=1,
        de265_chroma_422=2,
        de265_chroma_444=3

    ctypedef int64_t de265_PTS

    cdef int de265_get_image_width(de265_image*, int channel)
    cdef int de265_get_image_height(de265_image*, int channel)
    cdef de265_chroma de265_get_chroma_format(de265_image*)
    cdef int de265_get_bits_per_pixel(de265_image*, int channel)
    cdef uint8_t* de265_get_image_plane(de265_image*, int channel, int* out_stride)
    cdef void* de265_get_image_plane_user_data(de265_image*, int channel)
    # cdef de265_PTS de265_get_image_PTS(const struct de265_image*)
    cdef void* de265_get_image_user_data(de265_image*)
    cdef void de265_set_image_user_data(de265_image*, void *user_data)

    cdef void de265_get_image_NAL_header(de265_image*, int* nal_unit_type,
                                         char** nal_unit_name, int* nuh_layer_id,
                                         int* nuh_temporal_id)
    cdef de265_decoder_context* de265_new_decoder()
    cdef de265_error de265_start_worker_threads(de265_decoder_context*, int number_of_threads)
    cdef de265_error de265_free_decoder(de265_decoder_context*)

    cdef de265_error de265_push_data(de265_decoder_context*, const void* data, int length,
                                         de265_PTS pts, void* user_data)

    cdef void de265_push_end_of_NAL(de265_decoder_context*)

    cdef void de265_push_end_of_frame(de265_decoder_context*)

    cdef de265_error de265_push_NAL(de265_decoder_context*, const void* data, int length,
                                        de265_PTS pts, void* user_data)

    cdef de265_error de265_flush_data(de265_decoder_context*)

    cdef int de265_get_number_of_input_bytes_pending(de265_decoder_context*)

    cdef int de265_get_number_of_NAL_units_pending(de265_decoder_context*)

    cdef de265_error de265_decode(de265_decoder_context*, int* more)
    cdef void de265_reset(de265_decoder_context*)
    cdef de265_image* de265_peek_next_picture(de265_decoder_context*)
    cdef de265_image* de265_get_next_picture(de265_decoder_context*)
    cdef void de265_release_next_picture(de265_decoder_context*)
    cdef de265_error de265_get_warning(de265_decoder_context*)


    enum de265_image_format:
        de265_image_format_mono8    = 1,
        de265_image_format_YUV420P8 = 2,
        de265_image_format_YUV422P8 = 3,
        de265_image_format_YUV444P8 = 4

    struct de265_image_spec:
        de265_image_format format;
        int width
        int height
        int alignment

        int crop_left
        int crop_right
        int crop_top
        int crop_bottom

        int visible_width
        int visible_height

    struct de265_image_allocation:
        int (*get_buffer)(de265_decoder_context* ctx,
                                de265_image_spec* spec,
                                de265_image* img,
                                void* userdata)
        void (*release_buffer)(de265_decoder_context* ctx,
                                    de265_image* img,
                                    void* userdata)

    cdef void de265_set_image_allocation_functions(de265_decoder_context*,
                                                   de265_image_allocation*,
                                                   void* userdata)
    cdef de265_image_allocation *de265_get_default_image_allocation_functions()

    cdef void de265_set_image_plane(de265_image* img, int cIdx, void* mem, int stride, void *userdata)

    cdef int  de265_get_highest_TID(de265_decoder_context*)
    cdef int  de265_get_current_TID(de265_decoder_context*)

    cdef void de265_set_limit_TID(de265_decoder_context*,int max_tid)
    cdef void de265_set_framerate_ratio(de265_decoder_context*,int percent)
    cdef int  de265_change_framerate(de265_decoder_context*,int more_vs_less)

    enum de265_param:
        DE265_DECODER_PARAM_BOOL_SEI_CHECK_HASH=0,
        DE265_DECODER_PARAM_DUMP_SPS_HEADERS=1,
        DE265_DECODER_PARAM_DUMP_VPS_HEADERS=2,
        DE265_DECODER_PARAM_DUMP_PPS_HEADERS=3,
        DE265_DECODER_PARAM_DUMP_SLICE_HEADERS=4,
        DE265_DECODER_PARAM_ACCELERATION_CODE=5,
        DE265_DECODER_PARAM_SUPPRESS_FAULTY_PICTURES=6,

        DE265_DECODER_PARAM_DISABLE_DEBLOCKING=7,
        DE265_DECODER_PARAM_DISABLE_SAO=8

    enum de265_acceleration:
        de265_acceleration_SCALAR = 0,
        de265_acceleration_MMX  = 10,
        de265_acceleration_SSE  = 20,
        de265_acceleration_SSE2 = 30,
        de265_acceleration_SSE4 = 40,
        de265_acceleration_AVX  = 50,
        de265_acceleration_AVX2 = 60,
        de265_acceleration_ARM  = 70,
        de265_acceleration_NEON = 80,
        de265_acceleration_AUTO = 10000

    cdef void de265_set_parameter_bool(de265_decoder_context*, de265_param param, int value)

    cdef void de265_set_parameter_int(de265_decoder_context*, de265_param param, int value)

    cdef int  de265_get_parameter_bool(de265_decoder_context*, de265_param param)
