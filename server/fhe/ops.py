"""HEIR-compiled CKKS dot product for 128-element face embeddings."""

from __future__ import annotations

import logging
from functools import lru_cache

from heir import compile

logger = logging.getLogger("fhe")

DOT_PRODUCT_MLIR = """
func.func @dot_product(%arg0: tensor<128xf32> {secret.secret}, %arg1: tensor<128xf32> {secret.secret}) -> f32 {
  %c0_f32 = arith.constant 0.0 : f32
  %0 = affine.for %arg2 = 0 to 128 iter_args(%iter = %c0_f32) -> (f32) {
    %1 = tensor.extract %arg0[%arg2] : tensor<128xf32>
    %2 = tensor.extract %arg1[%arg2] : tensor<128xf32>
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %iter, %3 : f32
    affine.yield %4 : f32
  }
  return %0 : f32
}
"""


@lru_cache(maxsize=1)
def fhe_dot_product_ctx():
    """Compile and set up the CKKS dot product. Cached — only runs once."""
    logger.info("Compiling CKKS dot product with HEIR...")
    ctx = compile(mlir_str=DOT_PRODUCT_MLIR, scheme="ckks")
    logger.info("HEIR compilation complete. Running key generation...")
    ctx.setup()
    logger.info("CKKS key generation complete. FHE ready.")
    return ctx
