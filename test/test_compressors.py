"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1].resolve()))

from fl_sim.compressors import Compressor, CompressorType


def test_compressors():
    compressor = Compressor()
    compressor.makeRandKCompressor(K=10, D=100)
    assert compressor.compressorType == CompressorType.RANDK_COMPRESSOR
    # TODO: add more tests
