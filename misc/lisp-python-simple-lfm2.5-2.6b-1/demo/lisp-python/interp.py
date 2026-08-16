#!/usr/bin/env python3
"""Simple Lisp interpreter in Python."""

import sys
from typing import Any, List, Union


class Symbol:
    """Represents a symbol (atom) in the Lisp language."""
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"Symbol({self.name!r})"


class Value:
    """Represents a value in the Lisp language."""
    def __init__(self, val=None):
        self.val = val

    def __eq__(self, other):
        if isinstance(other, Value):
            return self.val == other.val
        elif isinstance(other, str):
            return self.val == other
        else:
            return NotImplemented


def make_value(val: Any) -> Value:
    """Create a Value from an atom or number."""
    if isinstance(val, (int, float)):
        return Value(val)
    elif isinstance(val, str):
        # Check if it's a quoted symbol
        if val.startswith("'") and val.endswith("'"):
            return Symbol(val[1:-1])
        else:
            return Value(val)
    else:
        raise TypeError(f"Unsupported value type: {type(val)}")


def evaluate(expr: Union[str, List], env: dict = None) -> Any:
    """
    Evaluate a Lisp expression.
    
    Args:
        expr: An S-expression (string or list) representing the expression
        env: Environment mapping symbols to values
    
    Returns:
        The result of evaluating the expression
    """
    if env is None:
        env = {}
    
    # If it's a symbol, return it as a Value
    if isinstance(expr, str):
        if expr.startswith("'") and expr.endswith("'"):
            val = make_value(Symbol(expr[1:-1]))
            print(f"[evaluate] Symbol: {expr} -> {val}")
            return val
        else:
            # Non-quoted string - treat as a Value (not a symbol)
            val = make_value(expr)
            print(f"[evaluate] String: {expr} -> {val}")
            return val
    
    # If it's a list (S-expression), evaluate each element
    if isinstance(expr, list):
        # The first element is the function/operator
        func = evaluate(expr[0], env)
        print(f"[evaluate] List: {expr} -> func={func}, args={len(args)}")
        # Evaluate all arguments
        args = [evaluate(arg, env) for arg in expr[1:]]
        result = apply_function(func, args, env)
        print(f"[evaluate] Result: {result}")
        return result
    
    # Fallback: treat as a number
    try:
        return make_value(expr)
    except TypeError:
        raise ValueError(f"Unsupported expression type: {type(expr)}")


def apply_function(func: Any, args: List[Any], env: dict) -> Any:
    """Apply a function to its arguments."""
    if not isinstance(func, (list, tuple)) or len(func) == 0:
        raise ValueError("Function must be a non-empty list of symbols")
    
    # If it's a built-in function
    if func[0] in {'+', '-', '*', '/', '=', '<', '>', '<=', '>=', 'and', 'or', 'not'}:
        return apply_builtin(func, args, env)
    
    # Otherwise treat as a function name
    func_name = func[0]
    if func_name not in env:
        raise NameError(f"Unknown function: {func_name}")
    
    return env[func_name](*args, env=env)


def apply_builtin(func: List, args: List[Any], env: dict) -> Any:
    """Apply a built-in Lisp function."""
    if not func:
        raise ValueError("Built-in function requires arguments")
    
    # Unpack the first element as the function name
    fn = func[0]
    
    if fn == '+':
        return add(args, env)
    elif fn == '-':
        return sub(args, env)
    elif fn == '*':
        return mul(args, env)
    elif fn == '/':
        return div(args, env)
    elif fn == '=':
        return eq(args, env)
    elif fn == '<':
        return lt(args, env)
    elif fn == '>':
        return gt(args, env)
    elif fn == '<=':
        return leq(args, env)
    elif fn == '>=':
        return geq(args, env)
    elif fn == 'and':
        return and_(args, env)
    elif fn == 'or':
        return or_(args, env)
    elif fn == 'not':
        return not_(args, env)
    else:
        raise ValueError(f"Unknown built-in function: {fn}")


def add(args: List[Any], env: dict) -> Any:
    """Add multiple numbers."""
    if len(args) == 0:
        raise ValueError("add requires at least one argument")
    result = args[0]
    for arg in args[1:]:
        result = result + arg
    return result

def sub(args: List[Any], env: dict) -> Any:
    """Subtract multiple numbers."""
    if len(args) == 0:
        raise ValueError("sub requires at least one argument")
    result = args[0]
    for arg in args[1:]:
        result = result - arg
    return result

def mul(args: List[Any], env: dict) -> Any:
    """Multiply multiple numbers."""
    if len(args) == 0:
        raise ValueError("mul requires at least one argument")
    result = args[0]
    for arg in args[1:]:
        result = result * arg
    return result

def div(args: List[Any], env: dict) -> Any:
    """Divide multiple numbers."""
    if len(args) == 0:
        raise ValueError("div requires at least one argument")
    # Only first two are used for division in a simple way
    result = args[0]
    for arg in args[1:]:
        if arg == 0:
            raise ZeroDivisionError("division by zero")
        result = result / arg
    return result

def eq(args: List[Any], env: dict) -> bool:
    """Check equality of values."""
    if len(args) != 2:
        raise ValueError("eq requires exactly two arguments")
    return args[0] == args[1]

def lt(args: List[Any], env: dict) -> bool:
    """Less than comparison."""
    if len(args) != 2:
        raise ValueError("lt requires exactly two arguments")
    return args[0] < args[1]

def gt(args: List[Any], env: dict) -> bool:
    """Greater than comparison."""
    if len(args) != 2:
        raise ValueError("gt requires exactly two arguments")
    return args[0] > args[1]

def leq(args: List[Any], env: dict) -> bool:
    """Less than or equal comparison."""
    if len(args) != 2:
        raise ValueError("leq requires exactly two arguments")
    return args[0] <= args[1]

def geq(args: List[Any], env: dict) -> bool:
    """Greater than or equal comparison."""
    if len(args) != 2:
        raise ValueError("geq requires exactly two arguments")
    return args[0] >= args[1]

def and_(args: List[Any], env: dict) -> bool:
    """Logical AND of all arguments."""
    if not args:
        raise ValueError("and requires at least one argument")
    result = True
    for arg in args:
        if not arg:
            return False
        result = result and arg
    return result

def or_(args: List[Any], env: dict) -> bool:
    """Logical OR of all arguments."""
    if not args:
        raise ValueError("or requires at least one argument")
    result = True
    for arg in args:
        if arg:
            return True
        result = result or arg
    return result

def not_(args: List[Any], env: dict) -> bool:
    """Logical NOT of all arguments."""
    if len(args) == 0:
        raise ValueError("not requires at least one argument")
    # If there's only one argument, it's the negation
    if len(args) == 1:
        return not args[0]
    else:
        # For multiple arguments, NOT is applied to each and then combined with AND
        result = True
        for arg in args:
            if not arg:
                return False
            result = result and not arg
        return result


def print_value(value: Any, env: dict) -> None:
    """Print a value to stdout."""
    # If it's a symbol, just print the name
    if isinstance(value, Symbol):
        print(f"(print {value.name})")
    else:
        # For numbers and other values, use repr for clarity
        print(repr(value))


def run_lisp_file(filepath: str) -> None:
    """Run a Lisp file and print the result."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Simple parser: find all S-expressions (lists starting with `(`)
    import re
    
    result = evaluate(content, env={})
    print(f"Result of {filepath}: {result}")


def main():
    """Main entry point for running Lisp files."""
    if len(sys.argv) < 2:
        print("Usage: python interp.py <lisp-file> [files...]")
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        try:
            run_lisp_file(filepath)
        except Exception as e:
            print(f"Error running {filepath}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
