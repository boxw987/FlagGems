import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def log10_func(x):
    return tl.log(x.to(tl.float32)) * 0.4342944819032518


def log10(A):
    logger.debug("GEMS LOG10")
    return log10_func(A)


def log10_(A):
    logger.debug("GEMS LOG10_")
    if A.is_contiguous():
        return log10_func(A, out0=A)

    buf = A.contiguous()
    log10_func(buf, out0=buf)
    A.copy_(buf)
    return A


def log10_out(A, out):
    logger.debug("GEMS LOG10_OUT")
    src = A if A.is_contiguous() else A.contiguous()
    if out.is_contiguous():
        return log10_func(src, out0=out)

    out_buf = torch.empty(out.shape, device=out.device, dtype=out.dtype)
    log10_func(src, out0=out_buf)
    out.copy_(out_buf)
    return out
