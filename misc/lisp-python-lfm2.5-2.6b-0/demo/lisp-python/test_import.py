#!/usr/bin/env python3
"""Test that the run function can be called directly without imports."""

import ast
from typing import Any, Dict, List, Union


class SymbolTable:
    def __init__(self):
        self.global_env: Dict[str, Any] = {}
        self.local_env: Dict[str, Any] = {}

    def get(self, var: str) -> Any:
        if var in self.global_env:
            return self.global_env[var]
        if var in self.local_env:
            return self.local_env[var]
        raise NameError(f"Variable '{var}' not found")

    def set(self, var: str, value: Any) -> None:
        self.local_env[var] = value


class LispInterpreter:
    ARITHMETIC = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y if y != 0 else ZeroDivisionError("division by zero"),
    }

    LOGICAL = {
        'and': lambda a, b: a and b,
        'or': lambda a, b: a or b,
        'not': lambda a: not a,
        '=': lambda a, b: a == b,
    }

    def evaluate(self, expr: str) -> Any:
        code = expr.strip()
        if not code:
            return None

        # Find matching parentheses for nested expressions
        depth = 0
        start = 0
        i = 0
        while i < len(code):
            ch = code[i]
            if ch == '(':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    break
            i += 1

        inner = code[start+1:i]

        # Try to evaluate as a number first
        try:
            return float(inner)
        except ValueError:
            pass

        # Otherwise, parse as S-expression: (operator arg1 arg2 ...)
        if not inner.startswith('(') or not inner.endswith(')'):
            raise SyntaxError(f"Invalid expression: {expr}")

        content = inner[1:-1]
        # Find the operator (first element after '(')
        for i, ch in enumerate(content):
            if ch == '+':
                args = []
                j = i + 1
                while j < len(content) and content[j:i] != '':
                    try:
                        args.append(self.evaluate(content[j:]))
                    j += 1
                break
        else:
            raise SyntaxError(f"Invalid expression: {expr}")

        # Evaluate arguments
        result = None
        for arg in args:
            if isinstance(arg, str):
                try:
                    result = self.evaluate(arg)
                except Exception as e:
                    raise ValueError(f"Error evaluating argument '{arg}': {e}")
            else:
                pass
        return result


def run(expr: str) -> Any:
    interpreter = LispInterpreter()
    return interpreter.evaluate(expr)


if __name__ == "__main__":
    # Test simple arithmetic expressions
    result = run("(+ 1 2)")
    print(f"Result of (+ 1 2): {result}")

    result = run("(* 3 4)")
    print(f"Result of (* 3 4): {result}")
