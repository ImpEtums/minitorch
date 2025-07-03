from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    
    # Calculate new dimensions
    new_height = height // kh
    new_width = width // kw
    
    # Reshape input to group kernel-sized blocks
    # First reshape to separate the kernel dimensions
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    
    # Permute to bring kernel dimensions together
    permuted = reshaped.permute(0, 1, 2, 4, 3, 5)
    
    # Final reshape to flatten kernel dimensions
    tiled = permuted.contiguous().view(batch, channel, new_height, new_width, kh * kw)
    
    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.
    kh, kw = kernel
    
    # Calculate output dimensions
    new_height = height // kh
    new_width = width // kw
    
    # Use tile function to reshape input for pooling
    tiled, _, _ = tile(input, kernel)
    
    # Sum over the kernel dimension and then divide by kernel size
    # tiled shape: (batch, channel, new_height, new_width, kh * kw)
    summed = tiled.sum(dim=4)
    pooled = (summed / (kh * kw)).contiguous().view(batch, channel, new_height, new_width)
    
    return pooled


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        dim_val = int(dim.item())
        out = max_reduce(input, dim_val)
        ctx.save_for_backward(input, out, tensor(dim_val))
        ctx.dim = dim_val
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        input, out, dim_tensor = ctx.saved_values
        dim = ctx.dim
        
        # Create argmax mask
        argmax_mask = argmax(input, dim)
        
        # The gradient flows to all positions that achieved the maximum
        grad_input = argmax_mask * grad_output
        
        return grad_input, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, tensor(dim))


class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        dim_val = int(dim.item())
        
        # For numerical stability, subtract max before exponential
        max_vals = max(input, dim_val)
        shifted = input - max_vals
        
        # Compute exponentials
        exp_vals = shifted.exp()
        
        # Sum exponentials along dimension
        sum_exp = exp_vals.sum(dim_val)
        
        # Divide by sum to get softmax
        result = exp_vals / sum_exp
        
        ctx.save_for_backward(result, tensor(dim_val))
        ctx.dim = dim_val
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        softmax_output, _ = ctx.saved_values
        dim = ctx.dim
        
        # Gradient of softmax: grad_input = softmax * (grad_output - (softmax * grad_output).sum(dim))
        # This is the standard softmax gradient formula
        grad_sum = (softmax_output * grad_output).sum(dim)
        grad_input = softmax_output * (grad_output - grad_sum)
        
        return grad_input, 0.0


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.

    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    return Softmax.apply(input, tensor(dim))


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        dim_val = int(dim.item())
        
        # For numerical stability, use the log-sum-exp trick
        max_vals = max(input, dim_val)
        shifted = input - max_vals
        
        # Compute log(sum(exp(shifted_vals)))
        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(dim_val)
        log_sum_exp = sum_exp.log()
        
        # Result is shifted - log_sum_exp
        result = shifted - log_sum_exp
        
        ctx.save_for_backward(exp_shifted, sum_exp, tensor(dim_val))
        ctx.dim = dim_val
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        exp_shifted, sum_exp, _ = ctx.saved_values
        dim = ctx.dim
        
        # Gradient of log-softmax: grad_input = grad_output - softmax * grad_output.sum(dim)
        # where softmax = exp_shifted / sum_exp
        softmax_vals = exp_shifted / sum_exp
        grad_sum = grad_output.sum(dim)
        grad_input = grad_output - softmax_vals * grad_sum
        
        return grad_input, 0.0


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    return LogSoftmax.apply(input, tensor(dim))


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    
    # Use tile function to reshape input for pooling
    tiled, new_height, new_width = tile(input, kernel)
    
    # Take max over the kernel dimension (last dimension)
    # tiled shape: (batch, channel, new_height, new_width, kh * kw)
    pooled = max(tiled, dim=4).contiguous().view(batch, channel, new_height, new_width)
    
    return pooled


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    if ignore:
        return input
    
    # Generate random tensor with same shape as input
    random_tensor = rand(input.shape, backend=input.backend)
    
    # Create dropout mask: 1 where we keep values, 0 where we drop
    keep_mask = random_tensor > rate
    
    # Apply mask and scale by (1 / (1 - rate)) to maintain expected value
    if rate == 0.0:
        return input
    elif rate == 1.0:
        # Drop everything
        return input * 0.0
    else:
        return input * keep_mask / (1.0 - rate)
