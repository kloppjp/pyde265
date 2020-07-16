from enum import IntEnum


class ChromaFormat(IntEnum):
    MONO = 0
    C420 = 1
    C422 = 2
    C444 = 3


class InternalSignal(IntEnum):
    PREDICTION = 0
    RESIDUAL = 1
    TR_COEFF = 2
