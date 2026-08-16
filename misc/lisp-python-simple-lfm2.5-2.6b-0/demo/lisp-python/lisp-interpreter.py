#!/usr/bin/env python3
"""Simple Lisp interpreter in Python."""

import os
import sys
from typing import Any, List

def evaluate(expr: Any) -> Any:
    """Evaluate a Lisp expression."""
    if isinstance(expr, list):
        # S-expression: apply the first element as function to the rest
        func = expr[0]
        args = expr[1:] if len(expr) > 1 else []
        return _eval_func(func, *args)
    elif isinstance(expr, (int, float)):
        return expr
    elif isinstance(expr, str):
        # Atom - could be a number or symbol
        try:
            return int(expr)
        except ValueError:
            pass
        try:
            return float(expr)
        except ValueError:
            pass
        if expr.lower() in ('true', 'false'):
            return expr.lower() == 'true'
    elif isinstance(expr, bool):
        return expr
    else:
        raise TypeError(f"Unsupported expression type: {type(expr)}")

def _eval_func(func, *args):
    """Evaluate a function call with given arguments."""
    if func == 'print':
        result = evaluate(args[0])
        print(result)
        return None
    elif isinstance(func, str):
        # Built-in functions: arithmetic and logical
        ops = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y if y != 0 else ZeroDivisionError("division by zero"),
            'and': lambda a, b: bool(a) and bool(b),
            'or': lambda a, b: bool(a) or bool(b),
            'not': lambda a: not bool(a),
        }
        if func in ops:
            return ops[func](*args)
        else:
            raise NameError(f"Unknown function: {func}")
    else:
        raise TypeError(f"Unsupported function: {func}")

def run_program(program):
    """Run a Lisp program and print the result."""
    result = evaluate(program)
    print(result)
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Try to read from file first
        path = sys.argv[1]
        if os.path.isfile(path):
            with open(path, 'r') as f:
                content = f.read().strip()
            # If it looks like a single S-expression (starts with [ or (, evaluate directly
            if content and (content.startswith('[') or content.startswith('(')):
                try:
                    program = eval(content)
                except Exception as e:
                    print(f"Error: {e}")
                    sys.exit(1)
            else:
                # Wrap in a list for consistent evaluation
                program = [content]
        else:
            # If not a file, treat the argument as an expression to evaluate directly
            try:
                program = eval(sys.argv[1])
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)
    else:
        # Interactive mode - read S-expressions one per line
        print("Lisp Interpreter (type expressions, Ctrl-D to finish):")
        program = []
    run_program(program)
