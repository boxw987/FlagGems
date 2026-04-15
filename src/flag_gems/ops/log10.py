import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def log10_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    COMPUTE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)

    if COMPUTE_FP64:
        y = tl.log(x) * 0.4342944819032518
    else:
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518

    tl.store(out_ptr + offsets, y, mask=mask)


def _log10_expected_dtype(x):
    return x.dtype if x.is_floating_point() else torch.float32


def _launch_log10(in_tensor, out_tensor):
    n_elements = in_tensor.numel()
    if n_elements == 0:
        return

    compute_fp64 = in_tensor.dtype == torch.float64
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(in_tensor.device):
        log10_kernel[grid](
            in_tensor,
            out_tensor,
            n_elements,
            COMPUTE_FP64=compute_fp64,
            BLOCK_SIZE=1024,
        )


def log10(A):
    logger.debug("GEMS LOG10")
    if not A.is_cuda:
        return torch.log10(A)

    out_dtype = _log10_expected_dtype(A)
    src = A if A.is_contiguous() else A.contiguous()
    out = torch.empty(A.shape, device=A.device, dtype=out_dtype)
    _launch_log10(src, out)
    return out


def log10_(A):
    logger.debug("GEMS LOG10_")
    if not A.is_cuda:
        return torch.log10_(A)
    if not A.is_floating_point():
        raise TypeError("log10_ only supports floating point tensors")

    if A.is_contiguous():
        _launch_log10(A, A)
        return A

    buf = A.contiguous()
    _launch_log10(buf, buf)
    A.copy_(buf)
    return A


def log10_out(A, out):
    logger.debug("GEMS LOG10_OUT")
    if not A.is_cuda:
        return torch.log10(A, out=out)

    expected_dtype = _log10_expected_dtype(A)
    if out.shape != A.shape:
        raise ValueError("log10.out expects out to have the same shape as input")
    if out.dtype != expected_dtype:
        raise TypeError(
            f"log10.out expects out dtype {expected_dtype}, but got {out.dtype}"
        )

    src = A if A.is_contiguous() else A.contiguous()
    if out.is_contiguous():
        _launch_log10(src, out)
        return out

    out_buf = torch.empty(out.shape, device=out.device, dtype=out.dtype)
    _launch_log10(src, out_buf)
    out.copy_(out_buf)
    return out
