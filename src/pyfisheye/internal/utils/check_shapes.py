from functools import wraps
from ndshapecheck import ShapeCheck # type: ignore
from typing import Any
import inspect

def check_shapes(param_names_and_rules: dict[str, str]):
    """
    A decorator for checking the shapes of input elements. 

    :param param_names_and_rules: A dictionary containing each parameter to be checked and the
        corresponding rule. Only objects having 'shape' attributes are checked.
    :returns: The decorator.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(fn)
            bound_args = sig.bind(*args, **kwargs)
            param_mapping = dict(bound_args.arguments)
            sc = ShapeCheck()
            for param_name, param_value in param_mapping.items():
                if param_name in param_names_and_rules and hasattr(param_value, 'shape'):
                    valid = sc(param_names_and_rules[param_name]).check(param_value)
                    if not valid:
                        raise ValueError(f"Invalid shape for '{param_name}': {sc.why}")
            return fn(*args, **kwargs)
        return wrapper
    return decorator
