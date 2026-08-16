#!/usr/bin/env python3
"""Simple Lisp interpreter in Python."""

import ast
import sys
from typing import Any, Dict, List, Union


class SymbolTable:
    """Environment for variable bindings."""
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
    """Simple Lisp interpreter with custom S-expression parser."""

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

    def evaluate(self, expr: Any) -> Any:
        """Evaluate a Lisp expression."""
        if isinstance(expr, list):
            return self.eval_list(expr)
        elif isinstance(expr, str):
            try:
                return float(expr)
            except ValueError:
                return expr
        else:
            raise TypeError(f"Unsupported expression type: {type(expr)}")

    def eval_list(self, lst: List[Any]) -> Any:
        """Evaluate a list (S-expression)."""
        if not lst:
            return None
        op = lst[0]
        args = lst[1:] if len(lst) > 1 else []
        return self.eval_operator(op, args)

    def eval_operator(self, op: str, args: List[Any]) -> Any:
        """Evaluate an operator with its arguments."""
        if op not in self.ARITHMETIC and op not in self.LOGICAL:
            raise ValueError(f"Unknown operator: {op}")

        # Arithmetic operators are variadic (can take any number of args)
        if op in self.ARITHMETIC:
            return self.ARITHMETIC[op](*args)  # unpack all arguments

        # Logical operators require exactly 2 arguments
        if op == 'and' or op == 'or':
            if len(args) < 2:
                raise ValueError(f"Operator {op} requires at least 2 arguments")
            return self.LOGICAL[op](args[0], args[1])
        elif op == 'not':
            return self.LOGICAL['not'](args[0])
        else:
            if len(args) != 2:
                raise ValueError(f"Operator {op} expects exactly 2 arguments, got {len(args)}")
            a = self.evaluate(args[0])
            b = self.evaluate(args[1])
            return self.ARITHMETIC[op](a, b)

    def while_loop(self, cond: Any, body: List[Any]) -> Any:
        """Evaluate a while loop."""
        result = self.evaluate(cond)
        while result:
            for arg in body:
                result = self.evaluate(arg)
        return result

    def if_statement(self, cond: Any, then_branch: List[Any], else_branch: List[Any]) -> Any:
        """Evaluate an if statement."""
        cond_val = self.evaluate(cond)
        if cond_val:
            return self.eval_list(then_branch)
        else:
            return self.eval_list(else_branch)


def run_lisp(code: str, env: SymbolTable = None) -> Any:
    """Run a Lisp program and return the result."""
    if env is None:
        env = SymbolTable()
    try:
        tree = parse_s_expr(code)
    except SyntaxError as e:
        raise ValueError(f"Invalid Lisp syntax: {e}")
    
    result = None
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            result = this_eval_node(node.value)
        elif isinstance(node.value, (ast.BinOp, ast.UnaryOp, ast.Name)):
            # Numbers are handled directly by this_eval_node
            pass
    return result

def parse_s_expr(code: str) -> Union[List[Any], Any]:
    """Parse a Lisp expression string into an AST."""
    code = code.strip()
    if not code:
        return None
    
    # If it starts with '(', it's a list (S-expression)
    if code.startswith('('):
        # Find matching closing parenthesis
        depth = 0
        i = 1
        while i < len(code) and depth > 0:
            if code[i] == '(':
                depth += 1
            elif code[i] == ')':
                depth -= 1
            i += 1
        
        # Extract content between outer parentheses (exclude the parens themselves)
        inner = code[1:i-1]
        if not inner:
            return None
        # Recursively parse the list
        return [parse_s_expr(inner)]
    else:
        # It's an atom: number or symbol
        try:
            return float(code)
        except ValueError:
            return code  # symbol

def this_eval_node(node: ast.AST) -> Any:
    """Evaluate a single AST node."""
    if isinstance(node, ast.List):
        return [this_eval_node(child) for child in node.elts]
    elif isinstance(node, ast.BinOp):
        left = this_eval_node(node.left)
        right = this_eval_node(node.right)
        op_type = type(node.op).__name__
        if op_type == 'Add':
            return left + right
        elif op_type == 'Sub':
            return left - right
        elif op_type == 'Mul':
            return left * right
        elif op_type == 'Div':
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
    elif isinstance(node, ast.UnaryOp):
        operand = this_eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return +operand
    elif isinstance(node, ast.Name):
        try:
            return env.get(node.id)
        except NameError:
            raise NameError(f"Variable '{node.id}' not found in environment")
    else:
        raise TypeError(f"Unsupported AST node: {type(node)}")


def run(code: str) -> Any:
    """Convenience function to run a Lisp expression."""
    return this_eval_node(parse_s_expr(code))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = run(sys.argv[1])
        print(result)
    else:
        print("Usage: python lisp-interpreter.py <lisp-expression>")