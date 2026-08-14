"""Simple Lisp interpreter in Python.

Implements a basic Lisp with:
- S-expression syntax (parentheses, atoms)
- Arithmetic operations (+, -, *, /)
- Function definitions and calls
- Quote for literal values
"""

import re
from typing import List, Union, Any


class Token:
    """A token in the Lisp source code."""
    def __init__(self, type: str, value: Any):
        self.type = type  # 'atom', 'operator', or 'quote'
        self.value = value

    def __repr__(self):
        return f"Token({self.type!r}, {self.value!r})"


class Lexer:
    """Lexer that converts Lisp source code into a list of tokens."""

    TOKENS = [
        ('atom', r'\d+'),
        ('symbol', r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('operator', r'[+\-*/=()]'),
        ('quote', r'\?'),
    ]

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.tokens = []

    def tokenize(self) -> List[Token]:
        """Convert the source code into a list of tokens."""
        i = 0
        while i < len(self.source):
            ch = self.source[i]

            # Skip whitespace
            if ch.isspace():
                i += 1
                continue

            # Number (non-negative integer)
            if ch.isdigit():
                start = i
                while i < len(self.source) and self.source[i].isdigit():
                    i += 1
                num = int(self.source[start:i])
                self.tokens.append(Token('atom', num))
                continue

            # Symbol (identifier or variable name)
            if ch.isalpha() or ch == '_':
                start = i
                while i < len(self.source) and (
                    self.source[i].isalnum() or self.source[i] == '_'
                ):
                    i += 1
                sym = self.source[start:i]
                self.tokens.append(Token('symbol', sym))
                continue

            # Operators and parentheses
            if ch in '(+\-*/=)':
                self.tokens.append(Token('operator', ch))
                i += 1
                continue
            if ch == '?':
                self.tokens.append(Token('quote', None))
                i += 1
                continue

            # Unknown character - skip or raise error
            i += 1

        return self.tokens


class Parser:
    """Parser that converts tokens into an Abstract Syntax Tree (AST)."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected_type: str = None) -> Token:
        """Consume the next token and optionally check its type."""
        token = self.peek()
        if token is None:
            raise SyntaxError("Unexpected end of input")
        if expected_type and token.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {token.type}")
        self.pos += 1
        return token

    def parse(self) -> List[Any]:
        """Parse the entire token stream into an AST."""
        result = []
        while self.peek() is not None:
            token = self.peek()

            if token.type == 'atom':
                # Atom: number or symbol (quoted)
                result.append(self.consume('atom'))
            elif token.type == 'operator':
                op = token.value
                # Check if this operator starts a function call
                # A function call has the form (op arg1 arg2 ...)
                # If we're at the start of parsing and there are more tokens after it,
                # and those tokens include atoms/operators, then it could be a nested call.
                # But for simplicity, handle grouping parentheses explicitly first.

                if op in '(+\-*/=':  # operators that can start calls
                    # Consume the operator and parse all following atoms/operators until ')'
                    result.append(('call', op, [self.consume('atom') for _ in range(len(self.tokens) - 1)]))
                else:
                    # Not a function call start (e.g., + as an element of a list)
                    pass
            elif token.type == 'quote':
                quote_val = self.consume('atom')
                if quote_val is None:
                    raise SyntaxError("Expected value after quote")
                result.append(('quote', quote_val.value))
            else:
                raise SyntaxError(f"Unexpected token: {token.type}")
        return result


class Evaluator:
    """Evaluates an AST (Abstract Syntax Tree) of Lisp expressions."""

    def __init__(self):
        self.globals = {}

    def set_var(self, name: str, value: Any):
        """Set a variable's value (set! macro)."""
        self.globals[name] = value

    def evaluate(self, expr: Any) -> Any:
        """Evaluate a single expression and return its value."""
        if isinstance(expr, tuple):
            # A call: ('call', operator, [args])
            op = expr[0]
            args = expr[2]
            if not args:
                raise SyntaxError("Function call with no arguments")
            return self.call(op, args)
        elif isinstance(expr, str):
            # A symbol (variable reference)
            return self.var_ref(expr)
        else:
            # An atom: number or quoted value
            if expr is None:
                raise SyntaxError("Unexpected null value")
            return expr

    def call(self, op: str, args: List[Any]) -> Any:
        """Evaluate a function call."""
        # Built-in functions
        builtins = {
            '+}': lambda *a: sum(a),
            '-': lambda *a: a[0] - a[1],
            '*' : lambda *a: eval_prod(a),
            '/': lambda *a: eval_div(a),
        }

        if op in builtins:
            return builtins[op](args)

        # Function definition (lambda) - handled during parsing
        raise NameError(f"Unknown function: {op}")

    def var_ref(self, name: str) -> Any:
        """Look up a variable and return its value."""
        if name in self.globals:
            return self.globals[name]
        # Try to evaluate as an expression (for dynamic binding)
        try:
            val = self.evaluate(name)
            self.globals[name] = val
            return val
        except SyntaxError:
            raise NameError(f"Undefined variable: {name}")

    def eval_prod(self, args: List[Any]) -> Any:
        """Evaluate a product (multiplication)."""
        if not args:
            raise ValueError("Product of empty list")
        result = 1
        for arg in args:
            result *= arg
        return result

    def eval_div(self, args: List[Any]) -> Any:
        """Evaluate a division."""
        if not args:
            raise ValueError("Division of empty list")
        result = 1
        for arg in reversed(args):
            if arg == 0:
                raise ZeroDivisionError("Division by zero")
            result /= arg
        return result


def run_lisp(source: str) -> Any:
    """Run a Lisp expression from source code and return the result."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    evaluator = Evaluator()
    return evaluator.evaluate(ast)
