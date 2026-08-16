#!/usr/bin/env python3
"""Simple Lisp interpreter in Python."""

import os
import sys
from typing import Any, List

def evaluate(expr):
    if isinstance(expr, list):
        func = expr[0]
        args = expr[1:] if len(expr) > 1 else []
        return _eval_func(func, *args)
    elif isinstance(expr, (int, float)):
        return expr
    elif isinstance(expr, str):
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
    if func == 'print':
        result = evaluate(args[0])
        print(result)
        return None
    elif isinstance(func, str):
        ops = {'+': lambda x, y: x + y,
              '-': lambda x, y: x - y,
              '*': lambda x, y: x * y,
              '/': lambda x, y: x / y if y != 0 else ZeroDivisionError("division by zero")}
        ops['and'] = lambda a, b: bool(a) and bool(b)
        ops['or'] = lambda a, b: bool(a) or bool(b)
        ops['not'] = lambda a: not bool(a)
        if func in ops:
            return ops[func](*args)
        else:
            raise NameError(f"Unknown function: {func}")
    else:
        raise TypeError(f"Unsupported function: {func}")

def run_program(program):
    result = evaluate(program)
    print(result)
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            with open(path, 'r') as f:
                content = f.read().strip()
            if content and (content.startswith('[') or content.startswith('(')):
                try:
                    program = eval(content)
                except Exception as e:
                    print(f"Error: {e}")
                    sys.exit(1)
            else:
                program = [content]
        else:
            try:
                program = eval(sys.argv[1])
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)
    else:
        print("Lisp Interpreter (type expressions, Ctrl-D to finish):")
        program = []
    run_program(program)
