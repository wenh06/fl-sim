import torch
from torch import _VF

from typing import Optional, Union, Tuple, List


def torch_norm(
    input: torch.Tensor,
    p: Optional[Union[float, str]] = "fro",
    dim: Optional[Union[int, Tuple[int], List[int]]] = None,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Returns the matrix norm or vector norm of a given tensor.

    Function copied and modified from PyTorch (v2.0) source code.

    .. warning::

        torch.norm is deprecated and may be removed in a future PyTorch release.
        Its documentation and behavior may be incorrect, and it is no longer
        actively maintained.

        Use :func:`torch.linalg.vector_norm` when computing vector norms and
        :func:`torch.linalg.matrix_norm` when computing matrix norms.
        For a function with a similar behavior as this one see :func:`torch.linalg.norm`.
        Note, however, the signature for these functions is slightly different than the
        signature for ``torch.norm``.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor. Its data type must be either a floating
        point or complex type. For complex inputs, the norm is calculated using the
        absolute value of each element. If the input is complex and neither
        :attr:`dtype` nor :attr:`out` is specified, the result's data type will
        be the corresponding floating point type (e.g. float if :attr:`input` is
        complexfloat).

    p : {int, float, inf, -inf, "fro", "nuc"}, optional
        the order of norm. Default: ``'fro'``
        The following norms can be calculated:

        ======  ==============  ==========================
        ord     matrix norm     vector norm
        ======  ==============  ==========================
        'fro'   Frobenius norm  --
        'nuc'   nuclear norm    --
        Number  --              sum(abs(x)**ord)**(1./ord)
        ======  ==============  ==========================

        The vector norm can be calculated across any number of dimensions.
        The corresponding dimensions of :attr:`input` are flattened into
        one dimension, and the norm is calculated on the flattened
        dimension.

        Frobenius norm produces the same result as ``p=2`` in all cases
        except when :attr:`dim` is a list of three or more dims, in which
        case Frobenius norm throws an error.

        Nuclear norm can only be calculated across exactly two dimensions.

    dim : int, tuple of ints, list of ints, optional
        Specifies which dimension or dimensions of :attr:`input` to
        calculate the norm across. If :attr:`dim` is ``None``, the norm will
        be calculated across all dimensions of :attr:`input`. If the norm
        type indicated by :attr:`p` does not support the specified number of
        dimensions, an error will occur.
    keepdim : bool, optional
        whether the output tensors have :attr:`dim`
        retained or not. Ignored if :attr:`dim` = ``None`` and
        :attr:`out` = ``None``. Default: ``False``
    out : torch.Tensor, optional
        the output tensor. Ignored if
        :attr:`dim` = ``None`` and :attr:`out` = ``None``.
    dtype : torch.dtype, optional
        the desired data type of
        returned tensor. If specified, the input tensor is casted to
        :attr:`dtype` while performing the operation. Default: None.

    .. note::
        Even though ``p='fro'`` supports any number of dimensions, the true
        mathematical definition of Frobenius norm only applies to tensors with
        exactly two dimensions. :func:`torch.linalg.matrix_norm` with ``ord='fro'``
        aligns with the mathematical definition, since it can only be applied across
        exactly two dimensions.

    Example
    -------
    >>> import torch
    >>> a = torch.arange(9, dtype= torch.float) - 4
    >>> b = a.reshape((3, 3))
    >>> torch.norm(a)
    tensor(7.7460)
    >>> torch.norm(b)
    tensor(7.7460)
    >>> torch.norm(a, float('inf'))
    tensor(4.)
    >>> torch.norm(b, float('inf'))
    tensor(4.)
    >>> c = torch.tensor([[ 1, 2, 3], [-1, 1, 4]] , dtype=torch.float)
    >>> torch.norm(c, dim=0)
    tensor([1.4142, 2.2361, 5.0000])
    >>> torch.norm(c, dim=1)
    tensor([3.7417, 4.2426])
    >>> torch.norm(c, p=1, dim=1)
    tensor([6., 6.])
    >>> d = torch.arange(8, dtype=torch.float).reshape(2, 2, 2)
    >>> torch.norm(d, dim=(1, 2))
    tensor([ 3.7417, 11.2250])
    >>> torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
    (tensor(3.7417), tensor(11.2250))

    """

    # if has_torch_function_unary(input):
    #     return handle_torch_function(
    #         norm, (input,), input, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype
    #     )

    # NB. All the repeated code and weird python is to please TorchScript.
    #     For a more compact implementation see the relevant function in `_refs/__init__.py`

    # We don't do this for MPS or sparse tensors
    if input.layout == torch.strided and input.device.type in ("cpu", "cuda", "meta"):
        if dim is not None:
            if isinstance(dim, int):
                _dim = [dim]
            else:
                _dim = dim
        else:
            _dim = None  # type: ignore[assignment]

        if isinstance(p, str):
            if p == "fro" and (dim is None or isinstance(dim, int) or len(dim) <= 2):
                if out is None:
                    return torch.linalg.vector_norm(
                        input, 2, _dim, keepdim, dtype=dtype
                    )
                else:
                    return torch.linalg.vector_norm(
                        input, 2, _dim, keepdim, dtype=dtype, out=out
                    )

            # Here we either call the nuclear norm, or we call matrix_norm with some arguments
            # that will throw an error
            if _dim is None:
                _dim = list(range(input.ndim))
            if out is None:
                return torch.linalg.matrix_norm(input, p, _dim, keepdim, dtype=dtype)
            else:
                return torch.linalg.matrix_norm(
                    input, p, _dim, keepdim, dtype=dtype, out=out
                )
        else:
            # NB. p should be Union[str, number], not Optional!
            _p = 2.0 if p is None else p
            if out is None:
                return torch.linalg.vector_norm(input, _p, _dim, keepdim, dtype=dtype)
            else:
                return torch.linalg.vector_norm(
                    input, _p, _dim, keepdim, dtype=dtype, out=out
                )

    ndim = input.dim()

    # catch default case
    if dim is None and out is None and dtype is None and p is not None:
        if isinstance(p, str):
            if p == "fro":
                return _VF.frobenius_norm(input, dim=(), keepdim=keepdim)
        if not isinstance(p, str):
            _dim = [
                i for i in range(ndim)
            ]  # noqa: C416 TODO: rewrite as list(range(m))
            return _VF.norm(input, p, dim=_dim, keepdim=keepdim)  # type: ignore[attr-defined]

    # TODO: when https://github.com/pytorch/pytorch/issues/33782 is fixed
    # remove the overloads where dim is an int and replace with BraodcastingList1
    # and remove next four lines, replace _dim with dim
    if dim is not None:
        if isinstance(dim, int):
            _dim = [dim]
        else:
            _dim = dim
    else:
        _dim = None  # type: ignore[assignment]

    if isinstance(p, str):
        if p == "fro":
            if dtype is not None:
                raise ValueError("dtype argument is not supported in frobenius norm")

            if _dim is None:
                _dim = list(range(ndim))
            if out is None:
                return _VF.frobenius_norm(input, _dim, keepdim=keepdim)
            else:
                return _VF.frobenius_norm(input, _dim, keepdim=keepdim, out=out)
        elif p == "nuc":
            if dtype is not None:
                raise ValueError("dtype argument is not supported in nuclear norm")
            if _dim is None:
                if out is None:
                    return _VF.nuclear_norm(input, keepdim=keepdim)
                else:
                    return _VF.nuclear_norm(input, keepdim=keepdim, out=out)
            else:
                if out is None:
                    return _VF.nuclear_norm(input, _dim, keepdim=keepdim)
                else:
                    return _VF.nuclear_norm(input, _dim, keepdim=keepdim, out=out)
        raise RuntimeError(f"only valid string values are 'fro' and 'nuc', found {p}")
    else:
        if _dim is None:
            _dim = list(range(ndim))

        if out is None:
            if dtype is None:
                return _VF.norm(input, p, _dim, keepdim=keepdim)  # type: ignore[attr-defined]
            else:
                return _VF.norm(input, p, _dim, keepdim=keepdim, dtype=dtype)  # type: ignore[attr-defined]
        else:
            if dtype is None:
                return _VF.norm(input, p, _dim, keepdim=keepdim, out=out)  # type: ignore[attr-defined]
            else:
                return _VF.norm(input, p, _dim, keepdim=keepdim, dtype=dtype, out=out)  # type: ignore[attr-defined]
