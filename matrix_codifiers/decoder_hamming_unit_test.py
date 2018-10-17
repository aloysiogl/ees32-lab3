import numpy as np
from .DecoderHamming import DecoderHamming

code = np.array([0, 0, 1, 1, 0, 0, 0], dtype=bool)
decoder = DecoderHamming()
print(decoder.decode(code).astype(np.uint8))
