from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals_plus = list(vals)
    vals_minus = list(vals)
    
    # Add epsilon to the arg-th position
    vals_plus[arg] += epsilon
    # Subtract epsilon from the arg-th position  
    vals_minus[arg] -= epsilon
    
    # Calculate central difference: (f(x+h) - f(x-h)) / (2*h)
    return (f(*vals_plus) - f(*vals_minus)) / (2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    visited = set()
    topo_order = []
    
    def dfs(var):
        if var.is_constant() or var.unique_id in visited:
            return
        visited.add(var.unique_id)
        
        # Visit parents first
        for parent in var.parents:
            if parent is not None:
                dfs(parent)
        
        topo_order.append(var)
    
    # Start DFS from the given variable
    dfs(variable)
    
    return topo_order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Build topological order using DFS
    visited = set()
    topo_order = []
    
    def dfs(var):
        if var.unique_id in visited:
            return
        visited.add(var.unique_id)
        
        if var.history is not None and var.history.inputs:
            for input_var in var.history.inputs:
                if input_var is not None:
                    dfs(input_var)
        
        topo_order.append(var)
    
    # Build topological order starting from output
    dfs(variable)
    
    # Initialize derivatives dictionary
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    
    # Process nodes in reverse topological order
    for var in reversed(topo_order):
        if var.unique_id not in derivatives:
            continue
            
        d_output = derivatives[var.unique_id]
        
        # If this is a leaf node or constant, accumulate derivative
        if var.is_leaf():
            var.accumulate_derivative(d_output)
            continue
        
        # Skip constants (no history)
        if var.history is None:
            continue
        
        # Get backward function and compute gradients
        last_fn = var.history.last_fn
        ctx = var.history.ctx
        inputs = var.history.inputs
        
        if inputs and last_fn is not None:
            # Call backward to get gradients
            gradients = last_fn._backward(ctx, d_output)
            
            # Handle single gradient case
            if not isinstance(gradients, tuple):
                gradients = (gradients,)
            
            # Accumulate gradients for each input
            for input_var, grad in zip(inputs, gradients):
                if input_var is not None:
                    key = input_var.unique_id
                    if key in derivatives:
                        derivatives[key] = derivatives[key] + grad
                    else:
                        derivatives[key] = grad


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
