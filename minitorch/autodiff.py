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
    vals_list = list(vals)
    
    # Create two copies of the arguments
    # vals_plus: x_i + epsilon
    vals_plus = vals_list.copy()
    vals_plus[arg] = vals_list[arg] + epsilon
    
    # vals_minus: x_i - epsilon  
    vals_minus = vals_list.copy()
    vals_minus[arg] = vals_list[arg] - epsilon
    
    # Apply central difference formula: [f(x + h) - f(x - h)] / (2h)
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)
    
    return (f_plus - f_minus) / (2.0 * epsilon)


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
    # Result list to store the topological order
    result = []
    
    def visit(var: Variable) -> None:
        """Recursive helper function for DFS-based topological sort"""
        # Skip if already visited or if it's a constant
        if var.unique_id in visited or var.is_constant():
            return
            
        # Mark as visited
        visited.add(var.unique_id)
        
        # Visit all parent nodes first (DFS)
        for parent in var.parents:
            visit(parent)
        
        # Add current node to result after visiting all parents
        result.append(var)
    
    # Start DFS from the given variable
    visit(variable)
    
    # Return in reverse order (from right to left in computation graph)
    return reversed(result)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    
    # Get topological order of variables
    topo_order = topological_sort(variable)
    
    # Dictionary to store derivatives for each variable
    derivatives = {}
    
    # Initialize the derivative of the output variable
    derivatives[variable.unique_id] = deriv
    
    # Propagate derivatives backward through the computation graph
    for var in topo_order:
        # Skip if no derivative computed for this variable
        if var.unique_id not in derivatives:
            continue
            
        # Get the current derivative
        current_deriv = derivatives[var.unique_id]
        
        # If this is a leaf node, accumulate the derivative
        if var.is_leaf():
            var.accumulate_derivative(current_deriv)
        
        # If this variable has parents, compute their derivatives using chain rule
        if not var.is_constant() and var.history is not None and var.history.last_fn is not None:
            # Get derivatives for parent variables using chain rule
            parent_derivs = var.chain_rule(current_deriv)
            
            # Accumulate derivatives for parent variables
            for parent_var, parent_deriv in parent_derivs:
                if parent_var.unique_id not in derivatives:
                    derivatives[parent_var.unique_id] = 0.0
                derivatives[parent_var.unique_id] += parent_deriv


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
