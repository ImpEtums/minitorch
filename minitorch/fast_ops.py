from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # Check if tensors are stride-aligned for optimization
        if (
            len(out_strides) == len(in_strides) 
            and np.array_equal(out_strides, in_strides)
            and np.array_equal(out_shape, in_shape)
        ):
            # Fast path: tensors are stride-aligned, iterate directly over storage
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            # General path: use numpy buffers for indices as required by docstring
            for i in prange(len(out)):
                # Create numpy buffers for indices
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)
                in_index = np.zeros(MAX_DIMS, dtype=np.int32)
                
                # Convert flat index to multidimensional index using pure calculations
                # Avoid any variable modifications that Numba might interpret as loop variable changes
                for dim_idx in range(len(out_shape)):
                    # Calculate coordinate for this dimension using pure math
                    # Get the product of all later dimensions
                    divisor = 1
                    for later_dim in range(dim_idx + 1, len(out_shape)):
                        divisor *= out_shape[later_dim]
                    
                    # Get the product of all earlier dimensions  
                    prior_product = 1
                    for earlier_dim in range(dim_idx):
                        prior_product *= out_shape[earlier_dim]
                    
                    # Calculate this dimension's coordinate
                    out_index[dim_idx] = (i // divisor) % out_shape[dim_idx]
                
                # Broadcast out_index to in_index
                for dim_idx in range(len(out_shape)):
                    if dim_idx < len(in_shape):
                        if in_shape[dim_idx] == 1:
                            in_index[dim_idx] = 0
                        else:
                            in_index[dim_idx] = out_index[dim_idx]
                    else:
                        in_index[dim_idx] = 0
                
                # Calculate positions using index_to_position equivalent
                out_pos = 0
                in_pos = 0
                for dim_idx in range(len(out_shape)):
                    out_pos += out_index[dim_idx] * out_strides[dim_idx]
                for dim_idx in range(len(in_shape)):
                    in_pos += in_index[dim_idx] * in_strides[dim_idx]
                
                out[out_pos] = fn(in_storage[in_pos])

    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # Check if all tensors are stride-aligned for optimization
        if (
            len(out_strides) == len(a_strides) == len(b_strides)
            and np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
            and np.array_equal(out_shape, a_shape)
            and np.array_equal(out_shape, b_shape)
        ):
            # Fast path: all tensors are stride-aligned, iterate directly over storage
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            # General path: use numpy buffers for indices as required by docstring
            for i in prange(len(out)):
                # Create numpy buffers for indices
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)
                a_index = np.zeros(MAX_DIMS, dtype=np.int32)
                b_index = np.zeros(MAX_DIMS, dtype=np.int32)
                
                # Convert flat index to multidimensional index using pure calculations
                for dim_idx in range(len(out_shape)):
                    # Calculate coordinate for this dimension using pure math
                    # Get the product of all later dimensions
                    divisor = 1
                    for later_dim in range(dim_idx + 1, len(out_shape)):
                        divisor *= out_shape[later_dim]
                    
                    # Calculate this dimension's coordinate
                    out_index[dim_idx] = (i // divisor) % out_shape[dim_idx]
                
                # Broadcast indices
                for dim_idx in range(len(out_shape)):
                    if dim_idx < len(a_shape):
                        if a_shape[dim_idx] == 1:
                            a_index[dim_idx] = 0
                        else:
                            a_index[dim_idx] = out_index[dim_idx]
                    else:
                        a_index[dim_idx] = 0
                        
                    if dim_idx < len(b_shape):
                        if b_shape[dim_idx] == 1:
                            b_index[dim_idx] = 0
                        else:
                            b_index[dim_idx] = out_index[dim_idx]
                    else:
                        b_index[dim_idx] = 0
                
                # Calculate positions
                out_pos = 0
                a_pos = 0
                b_pos = 0
                for dim_idx in range(len(out_shape)):
                    out_pos += out_index[dim_idx] * out_strides[dim_idx]
                for dim_idx in range(len(a_shape)):
                    a_pos += a_index[dim_idx] * a_strides[dim_idx]
                for dim_idx in range(len(b_shape)):
                    b_pos += b_index[dim_idx] * b_strides[dim_idx]
                
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # Parallel over output elements - use numpy buffers for indices
        for i in prange(len(out)):
            # Create numpy buffer for index
            out_index = np.zeros(MAX_DIMS, dtype=np.int32)
            
            # Convert flat index to multidimensional index using pure calculations
            for dim_idx in range(len(out_shape)):
                # Calculate coordinate for this dimension using pure math
                # Get the product of all later dimensions
                divisor = 1
                for later_dim in range(dim_idx + 1, len(out_shape)):
                    divisor *= out_shape[later_dim]
                
                # Calculate this dimension's coordinate
                out_index[dim_idx] = (i // divisor) % out_shape[dim_idx]
            
            # Calculate out_pos
            out_pos = 0
            for dim_idx in range(len(out_shape)):
                out_pos += out_index[dim_idx] * out_strides[dim_idx]
            
            # Reduce along the specified dimension
            for k in range(a_shape[reduce_dim]):
                # Create a_index by copying out_index and setting reduce_dim
                a_index = np.zeros(MAX_DIMS, dtype=np.int32)
                for dim_idx in range(len(out_shape)):
                    a_index[dim_idx] = out_index[dim_idx]
                a_index[reduce_dim] = k
                
                # Calculate a_pos
                a_pos = 0
                for dim_idx in range(len(a_shape)):
                    a_pos += a_index[dim_idx] * a_strides[dim_idx]
                
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.
    # Get dimensions
    batch_size = out_shape[0]
    out_rows = out_shape[1]  # a_shape[-2]
    out_cols = out_shape[2]  # b_shape[-1] 
    inner_dim = a_shape[2]   # a_shape[-1] == b_shape[-2]
    
    # Parallel over batch and output rows
    for batch in prange(batch_size):
        for i in range(out_rows):
            for j in range(out_cols):
                # Calculate output position
                out_pos = (
                    batch * out_strides[0] + 
                    i * out_strides[1] + 
                    j * out_strides[2]
                )
                
                # Accumulate dot product
                acc = 0.0
                for k in range(inner_dim):
                    # Calculate a position
                    a_pos = (
                        batch * a_batch_stride + 
                        i * a_strides[1] + 
                        k * a_strides[2]
                    )
                    # Calculate b position  
                    b_pos = (
                        batch * b_batch_stride + 
                        k * b_strides[1] + 
                        j * b_strides[2]
                    )
                    acc += a_storage[a_pos] * b_storage[b_pos]
                
                out[out_pos] = acc


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
