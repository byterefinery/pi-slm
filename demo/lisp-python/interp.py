"""Simple Lisp interpreter in Python."""

import ast
import operator

# Operation tables
ARITH = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}
LOGICAL = {
    'and': lambda a, b: bool(a) and bool(b),
    'or': lambda a, b: bool(a) or bool(b),
    'not': lambda a: not bool(a),
}

# AST node types (as strings for simplicity)
NODE_TYPES = {
    'symbol': ast.Constant,  # Python constant
    'list': list,
    'if': lambda n: isinstance(n[0], list) and len(n) == 3,
    'while': lambda n: isinstance(n[0], list) and len(n) == 2,
    'defun': lambda n: isinstance(n[0], list) and len(n) == 1 and n[0][0] == 'defun',
}


class LispInterpreter:
    """Simple Lisp interpreter supporting arithmetic, logical, conditionals, loops, and functions."""

    def __init__(self):
        self.env = {}  # variable -> value (dynamic scoping)

    def tokenize(self, code):
        """Convert source code into a list of tokens."""
        # Remove comments
        lines = code.split('\n')
        cleaned = []
        for line in lines:
            # Remove comments (// or #)
            i = line.find('//') if '//' in line else line.find('#')
            if i != -1:
                line = line[:i]
            cleaned.append(line.strip())
        code = ' '.join(cleaned)

        # Tokenize: parentheses, symbols (atoms), numbers
        tokens = []
        i = 0
        while i < len(code):
            ch = code[i]
            if ch.isspace():
                i += 1
                continue
            if ch in '()+-*/':
                # Multi-character operators: +-, *, /, ** (power)
                if ch == '*':
                    if i + 1 < len(code) and code[i+1] == '*':
                        tokens.append('**')
                        i += 2
                        continue
                tokens.append(ch)
                i += 1
            elif ch.isalpha() or ch == '_':
                # Read symbol (atom)
                j = i
                while j < len(code) and (code[j].isalnum() or code[j] == '_'):
                    j += 1
                tokens.append(code[i:j])
                i = j
            elif ch.isdigit():
                # Read number
                j = i
                while j < len(code) and code[j].isdigit():
                    j += 1
                tokens.append(int(code[i:j]))
                i = j
            else:
                # Unknown character - skip or raise
                i += 1
        return tokens

    def parse(self, code):
        """Parse S-expressions into an AST."""
        tokens = self.tokenize(code)
        if not tokens:
            raise SyntaxError("Empty input")
        # Parse as nested lists
        ast = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if isinstance(tok, list) and (len(tok) == 1 and tok[0] in ('if', 'while', 'defun')):
                # This is a form; parse its arguments
                args = self.parse_list(tok)
                ast.append(('form', args))
            elif isinstance(tok, list) and len(tok) == 2:
                # (cond body1 body2) for if
                cond = self.parse_list(tok[0])
                body = self.parse_list(tok[1]) if len(tok) > 1 else []
                ast.append(('if', [cond, body]))
            elif isinstance(tok, list) and len(tok) == 2:
                # (while cond do body)
                cond = self.parse_list(tok[0])
                body = self.parse_list(tok[1]) if len(tok) > 1 else []
                ast.append(('while', [cond, body]))
            elif isinstance(tok, list) and len(tok) == 1:
                # (defun name args body)
                defname = tok[0]
                args = self.parse_list(tok[1]) if len(tok) > 1 else []
                body = self.parse_list(tok[2]) if len(tok) > 2 else []
                ast.append(('defun', [defname, args, body]))
            elif isinstance(tok, list) and len(tok) == 3:
                # (operator arg1 ...)
                if tok[0] in ('+', '-', '*', '/'):
                    args = [self.evaluate(arg) for arg in tok[1:]]
                    ast.append(('apply', [tok[0], args]))
            elif isinstance(tok, list):
                # (form arg1 arg2 ...)
                args = self.parse_list(tok)
                ast.append(('apply', [tok[0], args]))
            else:
                raise SyntaxError(f"Unexpected token: {tok}")
            i += 1
        return ast

    def evaluate(self, ast):
        """Evaluate an AST to a value."""
        if not ast:
            return None

        node_type = ast[0]

        # Number (integer or float)
        if isinstance(ast[0], (int, float)):
            return ast[0]

        # Symbol (variable reference)
        if node_type == 'symbol':
            val = self.env.get(ast[1])
            if val is None:
                raise NameError(f"Undefined variable: {ast[1]}")
            return val

        # Form: (form arg1 arg2 ...)
        if node_type == 'apply':
            func = ast[1]
            args = [self.evaluate(arg) for arg in ast[2:]]
            # If the function is an arithmetic operator, apply it directly
            if isinstance(func, str) and func in ('+', '-', '*', '/'):
                return self.apply_arithmetic(func, args)
            # Otherwise treat as a function call
            if not isinstance(func, str):
                raise TypeError(f"Not a function: {func}")
            return self.apply_function(func, args)

        # Conditional: (if cond body1 body2)
        if node_type == 'if':
            cond = self.evaluate(ast[1])
            if cond:
                return self.evaluate(ast[2])
            else:
                return self.evaluate(ast[3])

        # While loop: (while cond do body)
        if node_type == 'while':
            while True:
                cond = self.evaluate(ast[1])
                if not cond:
                    break
                self.evaluate(ast[2])

        # Function definition: (defun name args body)
        if node_type == 'defun':
            defname = ast[1]
            args = [self.evaluate(arg) for arg in ast[2]]
            body = self.evaluate(ast[3])
            # Create a function object
            func = {
                'name': defname,
                'args': args,
                'body': body,
                'env': self.env.copy(),  # capture current env for closure
            }
            return func

        raise ValueError(f"Unknown node type: {node_type}")

    def apply_function(self, func, args):
        """Apply a function to its arguments."""
        if not isinstance(func, dict) or 'name' not in func:
            raise TypeError("Not a function")
        fn = func['name']
        arg_vals = [self.evaluate(arg) for arg in args]

        # Built-in functions
        if fn == '+':
            return ARITH[fn](*arg_vals)
        elif fn == '-':
            return ARITH['-'](*arg_vals)
        elif fn == '*':
            return ARITH['*'](*arg_vals)
        elif fn == '/':
            if not arg_vals:
                raise ZeroDivisionError("division by zero")
            return ARITH['/'](*arg_vals)
        # Logical operations
        elif fn == 'and':
            return LOGICAL['and'](*arg_vals)
        elif fn == 'or':
            return LOGICAL['or'](*arg_vals)
        elif fn == 'not':
            return LOGICAL['not'](arg_vals[0] if arg_vals else False)
        # Print function
        elif fn == 'print':
            for item in arg_vals:
                print(item)

        # Function call: (func arg1 ...)
        if fn in self.env:
            func_obj = self.env[fn]
            return self.apply_function(func_obj, arg_vals)
        elif isinstance(fn, str):
            # Built-in function lookup
            if fn == 'car':
                return lambda lst: lst[0] if lst else None
            elif fn == 'cdr':
                return lambda lst: lst[1:] if len(lst) > 1 else []

    def apply_arithmetic(self, func, args):
        """Apply an arithmetic operator to its arguments."""
        if not isinstance(func, str) or func not in ('+', '-', '*', '/'):
            raise ValueError(f"Unsupported operation: {func}")
        return self.apply_function(func, args)
        raise NameError(f"Unknown function: {fn}")


def run_lambda(code, env=None):
    """Run a Lisp expression and return the result."""
    interpreter = LispInterpreter()
    ast = interpreter.parse(code)
    if env is None:
        env = interpreter.env
    return interpreter.evaluate(ast)
