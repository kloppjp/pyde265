from pyde265 cimport de265, image, de265_internals
from pyde265.de265_enums import InternalSignal
from typing import Iterable, List, Union
from logging import getLogger
import array

_decoder_logger = getLogger("pyde265.Decoder")

cdef class Decoder(object):

    def __init__(self, n_threads: int = 4, buffer_size: int = 1024):
        self.n_threads = n_threads
        self.buffer_size = buffer_size
        self._context = de265.de265_new_decoder()
        self.signals_available = array.array('i', [0, 0, 0])
        _decoder_logger.info(f"Initialised with {n_threads} threads and {buffer_size}bytes buffer")
        de265.de265_start_worker_threads(self._context, self.n_threads)
        _decoder_logger.debug(f"Worker threads have been started")

    def __del__(self):
        de265.de265_free_decoder(self._context)
        _decoder_logger.debug(f"Decoder context has been released.")

    def activate_internal_signal(self, signal: InternalSignal):
        de265_internals.de265_internals_set_parameter_bool(self._context, signal.value, <int>1)
        self.signals_available[signal.value] = 1
        _decoder_logger.info(f"Activated internal signal {signal.name}")

    def deactivate_internal_signal(self, signal: InternalSignal):
        de265_internals.de265_internals_set_parameter_bool(self._context, signal.value, <int>0)
        self.signals_available[signal.value] = 0
        _decoder_logger.info(f"Deactivated internal signal {signal.name}")

    @property
    def available_signals(self) -> List[InternalSignal]:
        available_signals = list()
        if self.signals_available[0] == 1:
            available_signals.append(InternalSignal.PREDICTION)
        if self.signals_available[1] == 1:
            available_signals.append(InternalSignal.RESIDUAL)
        if self.signals_available[2] == 1:
            available_signals.append(InternalSignal.TR_COEFF)
        return available_signals

    def load_data(self, data):
        buffer = bytearray(self.buffer_size)
        cdef char * ba = buffer

        bytes_read = data.readinto(buffer)
        pos = 0
        cdef int user_data = 0
        # cdef de265.de265_error dec_error
        while bytes_read > 0:
            dec_error = de265.de265_push_data(self._context, ba, bytes_read, pos, &user_data)
            if dec_error != de265.DE265_OK:
                # ToDo: Throw exception?
                _decoder_logger.error(de265.de265_get_error_text(dec_error).decode("UTF-8", "replace"))
                break
            else:
                _decoder_logger.debug(f"No error during push data.")
            pos += bytes_read
            _decoder_logger.debug(f"Read {pos} bytes into buffer")
            bytes_read = data.readinto(buffer)

        dec_error = de265.de265_flush_data(self._context)
        if dec_error != de265.DE265_OK:
            # ToDo: Throw exception?
            _decoder_logger.error(de265.de265_get_error_text(dec_error).decode("UTF-8", "replace"))
        else:
            _decoder_logger.debug(f"No error during flush_data.")

    def decode_next_image(self) -> Union[image.Image, None]:
        cdef int more = 1
        while more > 0:
            more = 0

            dec_error = de265.de265_decode(self._context, &more)
            _decoder_logger.debug(f"Finished a decoding step. Is there more? {more}")
            if dec_error != de265.DE265_OK:
                # ToDo: Throw Exception?
                _decoder_logger.error(de265.de265_get_error_text(dec_error).decode("UTF-8", "replace"))
                break
            else:
                _decoder_logger.debug(f"No error during decode.")

            image_ptr = de265.de265_get_next_picture(self._context)

            if image_ptr == NULL:
                _decoder_logger.debug("Image pointer is null -> not yielding any image")
                continue
            result = image.Image.create(self._context, image_ptr)
            result.available_signals = self.available_signals
            _decoder_logger.info(f"Returning decoded image.")
            return result
        _decoder_logger.debug("Decoding has ended.")
        return None

    def decode(self, data) -> Iterable[image.Image]:
        self.load_data(data)

        next_image = self.decode_next_image()
        while next_image is not None:
            yield next_image
            next_image = self.decode_next_image()

    def free_image(self):
        _decoder_logger.debug("Freeing image.")
        de265.de265_release_next_picture(self._context)






