"""
Simple Lisp Interpreter in Python
==================================
Supports: arithmetic, logic, conditionals, loops, function definitions & calling.

Usage:
    python lisp.py                  # interactive REPL
    python lisp.py script.lisp      # run a .lisp file
    python -c "from lisp import Interpreter; Interpreter().eval(...)"  # programmatic
"""

import sys
import math
from typing import Any, Optional


# ── Exceptions ────────────────────────────────────────────────────────────────

class LispError(Exception):
    """Base exception for Lisp runtime errors."""


class LispEvalError(LispError):
    """Evaluation error."""


class LispReadError(LispError):
    """Parser/reader error."""


# ── Data Types ────────────────────────────────────────────────────────────────

class Symbol:
    """Lisp symbol (identifier)."""
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


class Nil:
    """Lisp nil (also serves as false and empty list)."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "nil"

    def __bool__(self):
        return False

    def __eq__(self, other):
        return isinstance(other, Nil)

    def __hash__(self):
        return 0


NIL = Nil()


class LispFunction:
    """A user-defined Lisp function."""
    __slots__ = ("params", "body", "env")

    def __init__(self, params: list, body: list, env: dict):
        self.params = params       # list of parameter symbols
        self.body = body           # list of expressions to evaluate
        self.env = env             # environment at definition time

    def __repr__(self):
        return f"#function({self.params})"


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(source: str) -> list:
    """
    Tokenize Lisp source code into a flat list of tokens.
    Handles: parentheses, quoted strings, numbers, symbols, comments.
    """
    tokens = []
    i = 0
    n = len(source)

    while i < n:
        ch = source[i]

        # skip whitespace
        if ch in " \t\n\r,":
            i += 1
            continue

        # skip comments
        if ch == ";":
            while i < n and source[i] != "\n":
                i += 1
            continue

        # parentheses (including square brackets, treated as parens)
        if ch == "(":
            tokens.append("(")
            i += 1
            continue
        if ch == ")":
            tokens.append(")")
            i += 1
            continue
        if ch == "[":
            tokens.append("(")
            i += 1
            continue
        if ch == "]":
            tokens.append(")")
            i += 1
            continue

        # quoted strings
        if ch == '"':
            j = i + 1
            while j < n and source[j] != '"':
                if source[j] == '\\' and j + 1 < n:
                    j += 2
                else:
                    j += 1
            if j >= n:
                raise LispReadError("Unterminated string")
            tokens.append(source[i:j + 1])
            i = j + 1
            continue

        # single-quoted strings (')
        if ch == "'":
            tokens.append("'")
            i += 1
            continue

        # backquote
        if ch == "`":
            tokens.append("`")
            i += 1
            continue

        # comma (unquote)
        if ch == ",":
            tokens.append(",")
            i += 1
            continue

        # atoms (numbers, symbols)
        j = i
        while j < n and source[j] not in " \t\n\r(),;\"'`[]":
            j += 1
        tokens.append(source[i:j])
        i = j

    return tokens


# ── Reader (Tokenizer → S-expressions) ────────────────────────────────────────

def read_sexpr(tokens: list, pos: int):
    """
    Read one S-expression from a token list starting at pos.
    Returns (sexpr, new_pos).
    """
    if pos >= len(tokens):
        raise LispReadError("Unexpected end of input")

    token = tokens[pos]

    if token == ")":
        raise LispReadError("Unexpected ')'")

    # string
    if token.startswith('"') and token.endswith('"'):
        # unescape
        s = token[1:-1].replace('\\"', '"').replace('\\\\', '\\').replace('\\n', '\n')
        return s, pos + 1

    # number
    try:
        if '.' in token:
            return float(token), pos + 1
        return int(token), pos + 1
    except ValueError:
        pass

    # symbol
    return Symbol(token), pos + 1


def read_all(tokens: list) -> list:
    """Read all S-expressions from a token list."""
    results = []
    pos = 0
    while pos < len(tokens):
        if tokens[pos] == ")":
            raise LispReadError("Unexpected ')'")
        sexpr, pos = _read_form(tokens, pos)
        results.append(sexpr)
    return results


def _read_form(tokens: list, pos: int):
    """Read one form (atom or list) from tokens."""
    if pos >= len(tokens):
        raise LispReadError("Unexpected end of input")

    token = tokens[pos]

    if token == ")":
        return None, pos

    # Handle ' (quote) prefix: '(1 2) → (quote (1 2))
    if token == "'":
        quoted_form, new_pos = _read_form(tokens, pos + 1)
        return [Symbol("quote"), quoted_form], new_pos

    if token == "(":
        # read list
        forms = []
        pos += 1  # skip '('
        while pos < len(tokens) and tokens[pos] != ")":
            form, pos = _read_form(tokens, pos)
            forms.append(form)
        if pos >= len(tokens):
            raise LispReadError("Unterminated list")
        return forms, pos + 1  # skip ')'

    # atom
    return read_sexpr(tokens, pos)


def read(source: str) -> list:
    """Read all S-expressions from a source string."""
    tokens = tokenize(source)
    return read_all(tokens)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Interpreter:
    """Lisp interpreter with environment-based variable lookup."""

    def __init__(self):
        # Built-in functions
        self._builtins = self._setup_builtins()
        # Global environment
        self.env = dict(self._builtins)
        # Hook for printing during evaluation
        self.output = []

    # ── Built-ins ─────────────────────────────────────────────────────────

    def _setup_builtins(self) -> dict:
        """Set up built-in functions and special values."""
        import operator

        def lisp_add(*args):
            if len(args) < 2:
                raise LispEvalError(f"+ expects at least 2 args, got {len(args)}")
            result = args[0]
            for a in args[1:]:
                if isinstance(result, str) or isinstance(a, str):
                    result = str(result) + str(a)
                else:
                    result = result + a
            return result

        def lisp_sub(*args):
            if len(args) < 1:
                raise LispEvalError("- expects at least 1 arg")
            if len(args) == 1:
                return -args[0]
            result = args[0]
            for a in args[1:]:
                result = result - a
            return result

        def lisp_mul(*args):
            if len(args) < 2:
                raise LispEvalError("* expects at least 2 args")
            result = args[0]
            for a in args[1:]:
                result = result * a
            return result

        def lisp_div(*args):
            if len(args) < 2:
                raise LispEvalError("/ expects at least 2 args")
            result = args[0]
            for a in args[1:]:
                if a == 0:
                    raise LispEvalError("Division by zero")
                result = result / a
            return result

        def lisp_mod(a, b):
            if b == 0:
                raise LispEvalError("Modulo by zero")
            return a % b

        def lisp_neg(x):
            return -x

        def lisp_abs(x):
            return abs(x)

        def lisp_max(*args):
            return max(args)

        def lisp_min(*args):
            return min(args)

        def lisp_sqrt(x):
            return math.sqrt(x)

        def lisp_int(x):
            return int(x)

        def lisp_float(x):
            return float(x)

        def lisp_print(*args):
            for arg in args:
                print(self._to_lisp_str(arg), end="")
            print()
            return NIL

        def lisp_println(*args):
            parts = []
            for arg in args:
                parts.append(self._to_lisp_str(arg))
            print(" ".join(parts))
            return NIL

        def lisp_input(prompt=""):
            if prompt:
                print(self._to_lisp_str(prompt), end="")
            return input()

        def lisp_car(lst):
            if not lst or not isinstance(lst, list):
                raise LispEvalError("car expects a non-empty list")
            return lst[0]

        def lisp_cdr(lst):
            if not lst or not isinstance(lst, list):
                raise LispEvalError("cdr expects a non-empty list")
            return lst[1:] if len(lst) > 1 else []

        def lisp_cons(a, lst):
            if not isinstance(lst, list):
                lst = []
            return [a] + lst

        def lisp_list(*args):
            return list(args)

        def lisp_len(lst):
            return len(lst)

        def lisp_reverse(lst):
            return list(reversed(lst))

        def lisp_nth(n, lst):
            if not isinstance(lst, list) or n < 0 or n >= len(lst):
                raise LispEvalError(f"nth: index {n} out of range for list of length {len(lst) if isinstance(lst, list) else '?'}")
            return lst[n]

        def lisp_concat(*lists):
            result = []
            for lst in lists:
                if isinstance(lst, list):
                    result.extend(lst)
                elif isinstance(lst, str):
                    result.append(lst)
                else:
                    result.append(lst)
            return result

        def lisp_range(*args):
            if len(args) == 1:
                return list(range(int(args[0])))
            elif len(args) == 2:
                return list(range(int(args[0]), int(args[1])))
            elif len(args) == 3:
                return list(range(int(args[0]), int(args[1]), int(args[2])))
            else:
                raise LispEvalError("range expects 1-3 args")

        def lisp_numberp(x):
            return x is not NIL and isinstance(x, (int, float))

        def lisp_stringp(x):
            return isinstance(x, str)

        def lisp_symbolp(x):
            return isinstance(x, Symbol)

        def lisp_listp(x):
            return isinstance(x, list)

        def lisp_nullp(x):
            return x is NIL or x == []

        def lisp_functionp(x):
            return isinstance(x, LispFunction) or callable(x)

        def lisp_type(x):
            if x is NIL:
                return "nil"
            if isinstance(x, bool):
                return "boolean"
            if isinstance(x, int):
                return "integer"
            if isinstance(x, float):
                return "float"
            if isinstance(x, str):
                return "string"
            if isinstance(x, Symbol):
                return "symbol"
            if isinstance(x, list):
                return "list"
            if isinstance(x, LispFunction):
                return "function"
            if callable(x):
                return "builtin"
            return "unknown"

        def lisp_str(*args):
            parts = []
            for a in args:
                parts.append(self._to_lisp_str(a))
            return "".join(parts)

        def lisp_string_to_number(s):
            try:
                return int(s)
            except (ValueError, TypeError):
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return 0

        builtins = {
            # Arithmetic
            "+": lisp_add,
            "-": lisp_sub,
            "*": lisp_mul,
            "/": lisp_div,
            "mod": lisp_mod,
            "neg": lisp_neg,
            "abs": lisp_abs,
            "max": lisp_max,
            "min": lisp_min,
            "sqrt": lisp_sqrt,
            "int": lisp_int,
            "float": lisp_float,

            # Comparison
            "<": operator.lt,
            ">": operator.gt,
            "<=": operator.le,
            ">=": operator.ge,
            "=": operator.eq,
            "!=": operator.ne,

            # Logic
            "not": lambda x: x is NIL or x is False,

            # I/O
            "print": lisp_print,
            "println": lisp_println,
            "input": lisp_input,

            # List operations
            "car": lisp_car,
            "cdr": lisp_cdr,
            "cons": lisp_cons,
            "list": lisp_list,
            "len": lisp_len,
            "reverse": lisp_reverse,
            "nth": lisp_nth,
            "concat": lisp_concat,
            "range": lisp_range,
            "empty?": lambda x: x is NIL or x == [],

            # Type predicates
            "number?": lisp_numberp,
            "string?": lisp_stringp,
            "symbol?": lisp_symbolp,
            "list?": lisp_listp,
            "nil?": lisp_nullp,
            "function?": lisp_functionp,
            "type": lisp_type,

            # String ops
            "str": lisp_str,
            "string->number": lisp_string_to_number,

            # Constants
            "true": True,
            "false": False,
            "nil": NIL,
        }

        return builtins

    # ── Helpers ───────────────────────────────────────────────────────────

    def _to_lisp_str(self, obj: Any) -> str:
        """Convert a Python object to its Lisp string representation."""
        if obj is NIL:
            return "nil"
        if obj is True:
            return "true"
        if obj is False:
            return "false"
        if isinstance(obj, Symbol):
            return obj.name
        if isinstance(obj, LispFunction):
            return "#function"
        if isinstance(obj, list):
            return "(" + " ".join(self._to_lisp_str(item) for item in obj) + ")"
        if callable(obj):
            return "#builtin"
        return str(obj)

    def _is_callable(self, obj: Any) -> bool:
        """Check if an object is callable (builtin or user function)."""
        return isinstance(obj, LispFunction) or callable(obj)

    def _call_builtin(self, func, *args):
        """Call a Python builtin function with Lisp args."""
        try:
            result = func(*args)
            # Convert Python bool to Lisp convention
            if result is True:
                return True
            if result is False:
                return False
            return result
        except LispError:
            raise
        except Exception as e:
            raise LispEvalError(f"Error calling builtin: {e}")

    def _call_function(self, func: Any, args: list):
        """Call either a builtin or user-defined function."""
        if isinstance(func, LispFunction):
            # Bind parameters in a new environment
            if len(args) != len(func.params):
                raise LispEvalError(
                    f"Function expects {len(func.params)} args, got {len(args)}"
                )
            new_env = dict(func.env)
            for param, arg in zip(func.params, args):
                new_env[param.name] = arg
            # Evaluate body expressions sequentially, return last
            result = NIL
            for expr in func.body:
                result = self.eval(expr, new_env)
            return result
        elif callable(func):
            return self._call_builtin(func, *args)
        else:
            raise LispEvalError(f"{self._to_lisp_str(func)} is not callable")

    # ── Core Evaluation ───────────────────────────────────────────────────

    def eval(self, expr: Any, env: Optional[dict] = None) -> Any:
        """
        Evaluate a Lisp expression.
        env: optional environment (defaults to global env).
        """
        if env is None:
            env = self.env

        # nil
        if expr is NIL:
            return NIL

        # boolean
        if isinstance(expr, bool):
            return expr

        # number
        if isinstance(expr, (int, float)):
            return expr

        # string
        if isinstance(expr, str):
            return expr

        # symbol → lookup in environment
        if isinstance(expr, Symbol):
            name = expr.name
            if name in env:
                return env[name]
            raise LispEvalError(f"Undefined variable: {name}")

        # list → special form or function call
        if isinstance(expr, list):
            if not expr:
                return []

            head = expr[0]

            # ── Special forms ───────────────────────────────────────────

            # quote: return the expression without evaluating
            if head == "quote" or head == "'":
                if len(expr) != 2:
                    raise LispEvalError("quote expects 1 argument")
                return expr[1]

            # if: conditional
            if head == "if":
                if len(expr) < 2 or len(expr) > 4:
                    raise LispEvalError("if expects 2-3 arguments: (if condition then [else])")
                condition = self.eval(expr[1], env)
                if condition is not NIL and condition is not False:
                    return self.eval(expr[2], env)
                elif len(expr) == 4:
                    return self.eval(expr[3], env)
                return NIL

            # cond: multi-way conditional
            if head == "cond":
                for clause in expr[1:]:
                    if not isinstance(clause, list) or len(clause) < 1:
                        raise LispEvalError(f"Invalid cond clause: {clause}")
                    # 'else' is a catch-all (always true)
                    if isinstance(clause[0], Symbol) and clause[0].name == "else":
                        if len(clause) == 1:
                            return NIL
                        return self.eval(clause[1], env)
                    test = self.eval(clause[0], env)
                    if test is not NIL and test is not False:
                        # Evaluate the result expression
                        if len(clause) == 1:
                            return test
                        return self.eval(clause[1], env)
                return NIL

            # def: define a global variable
            if head == "def":
                if len(expr) != 3:
                    raise LispEvalError("def expects 2 arguments: (def name value)")
                name = expr[1]
                if not isinstance(name, Symbol):
                    raise LispEvalError("def: name must be a symbol")
                value = self.eval(expr[2], env)
                self.env[name.name] = value
                return value

            # set!: reassign a variable
            if head == "set!":
                if len(expr) != 3:
                    raise LispEvalError("set! expects 2 arguments: (set! name value)")
                name = expr[1]
                if not isinstance(name, Symbol):
                    raise LispEvalError("set!: name must be a symbol")
                if name.name not in env:
                    raise LispEvalError(f"set!: undefined variable {name.name}")
                value = self.eval(expr[2], env)
                env[name.name] = value
                return value

            # defn: define a function
            if head == "defn":
                if len(expr) < 3:
                    raise LispEvalError("defn expects at least 2 arguments: (defn name [params] body...)")
                name = expr[1]
                if not isinstance(name, Symbol):
                    raise LispEvalError("defn: name must be a symbol")
                params = expr[2]
                if not isinstance(params, list):
                    raise LispEvalError("defn: params must be a list")
                param_symbols = []
                for p in params:
                    if isinstance(p, Symbol):
                        param_symbols.append(p)
                    else:
                        raise LispEvalError(f"defn: param must be a symbol, got {p}")
                body = expr[3:]
                func = LispFunction(param_symbols, body, self.env)
                self.env[name.name] = func
                return func

            # fn: anonymous function (lambda)
            if head == "fn" or head == "lambda":
                if len(expr) < 3:
                    raise LispEvalError("fn expects at least 2 arguments: (fn [params] body...)")
                params = expr[1]
                if not isinstance(params, list):
                    raise LispEvalError("fn: params must be a list")
                param_symbols = []
                for p in params:
                    if isinstance(p, Symbol):
                        param_symbols.append(p)
                    else:
                        raise LispEvalError(f"fn: param must be a symbol, got {p}")
                body = expr[2:]
                return LispFunction(param_symbols, body, env)

            # let: local variable bindings
            if head == "let":
                if len(expr) < 3:
                    raise LispEvalError("let expects bindings and body: (let [(name val) ...] body...)")
                bindings = expr[1]
                if not isinstance(bindings, list):
                    raise LispEvalError("let: bindings must be a list")
                new_env = dict(env)
                for binding in bindings:
                    if not isinstance(binding, list) or len(binding) != 2:
                        raise LispEvalError(f"let: invalid binding {binding}")
                    sym = binding[0]
                    if not isinstance(sym, Symbol):
                        raise LispEvalError("let: binding name must be a symbol")
                    val = self.eval(binding[1], env)
                    new_env[sym.name] = val
                # Evaluate body sequentially
                result = NIL
                for body_expr in expr[2:]:
                    result = self.eval(body_expr, new_env)
                return result

            # let*: sequential bindings (each can reference previous)
            if head == "let*":
                if len(expr) < 3:
                    raise LispEvalError("let* expects bindings and body")
                bindings = expr[1]
                new_env = dict(env)
                for binding in bindings:
                    if not isinstance(binding, list) or len(binding) != 2:
                        raise LispEvalError(f"let*: invalid binding {binding}")
                    sym = binding[0]
                    if not isinstance(sym, Symbol):
                        raise LispEvalError("let*: binding name must be a symbol")
                    val = self.eval(binding[1], new_env)  # new_env, not env!
                    new_env[sym.name] = val
                result = NIL
                for body_expr in expr[2:]:
                    result = self.eval(body_expr, new_env)
                return result

            # begin: sequential evaluation
            if head == "begin" or head == "do":
                result = NIL
                for body_expr in expr[1:]:
                    result = self.eval(body_expr, env)
                return result

            # while: loop
            if head == "while":
                if len(expr) < 2:
                    raise LispEvalError("while expects condition and body")
                condition = expr[1]
                body = expr[2:]
                result = NIL
                while True:
                    cond_val = self.eval(condition, env)
                    if cond_val is NIL or cond_val is False:
                        break
                    for body_expr in body:
                        result = self.eval(body_expr, env)
                return result

            # for: for-each style loop
            if head == "for":
                if len(expr) < 4:
                    raise LispEvalError("for expects: (for var list body...)")
                var = expr[1]
                if not isinstance(var, Symbol):
                    raise LispEvalError("for: var must be a symbol")
                lst = self.eval(expr[2], env)
                if not isinstance(lst, list):
                    raise LispEvalError("for: second arg must be a list")
                body = expr[3:]
                result = NIL
                for item in lst:
                    env[var.name] = item
                    for body_expr in body:
                        result = self.eval(body_expr, env)
                return result

            # when: if with implicit begin in then-branch
            if head == "when":
                if len(expr) < 2:
                    raise LispEvalError("when expects condition and body")
                condition = self.eval(expr[1], env)
                if condition is not NIL and condition is not False:
                    result = NIL
                    for body_expr in expr[2:]:
                        result = self.eval(body_expr, env)
                    return result
                return NIL

            # unless: inverse of when
            if head == "unless":
                if len(expr) < 2:
                    raise LispEvalError("unless expects condition and body")
                condition = self.eval(expr[1], env)
                if condition is NIL or condition is False:
                    result = NIL
                    for body_expr in expr[2:]:
                        result = self.eval(body_expr, env)
                    return result
                return NIL

            # and: short-circuit logical and
            if head == "and":
                result = True
                for arg in expr[1:]:
                    result = self.eval(arg, env)
                    if result is NIL or result is False:
                        return result
                return result

            # or: short-circuit logical or
            if head == "or":
                for arg in expr[1:]:
                    result = self.eval(arg, env)
                    if result is not NIL and result is not False:
                        return result
                return NIL

            # ── Function call ───────────────────────────────────────────

            # Evaluate the operator
            operator = self.eval(head, env)

            # Evaluate all arguments
            evaluated_args = [self.eval(arg, env) for arg in expr[1:]]

            # Call the function
            return self._call_function(operator, evaluated_args)

        # Anything else is a literal
        return expr

    # ── Top-level API ─────────────────────────────────────────────────────

    def evaluate(self, source: str) -> list:
        """Parse and evaluate a complete Lisp source string. Returns results."""
        forms = read(source)
        results = []
        for form in forms:
            results.append(self.eval(form))
        return results

    def evaluate_and_print(self, source: str):
        """Evaluate source and print the result of each form."""
        results = self.evaluate(source)
        for result in results:
            print(self._to_lisp_str(result))

    def repl(self):
        """Run an interactive read-eval-print loop."""
        print("Simple Lisp Interpreter")
        print("Type .quit or .exit to exit, .help for help")
        print()

        while True:
            try:
                line = input("lisp> ")
                if line.strip() in (".quit", ".exit", ".q"):
                    break
                if line.strip() == ".help":
                    self._print_help()
                    continue
                if line.strip() == ".env":
                    self._print_env()
                    continue
                result = self.evaluate(line)
                for r in result:
                    print(self._to_lisp_str(r))
            except EOFError:
                break
            except LispError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Internal error: {e}")

    def _print_help(self):
        print("""
Lisp Special Forms:
  (def name value)           - define a variable
  (set! name value)          - reassign a variable
  (defn name [params] body)  - define a function
  (fn [params] body)         - anonymous function
  (if cond then [else])      - conditional
  (cond [test result]...)    - multi-way conditional
  (let [(name val) ...] body) - local bindings
  (while condition body...)  - loop while true
  (for var list body...)     - iterate over list
  (begin expr...)            - sequential evaluation
  (quote expr) / 'expr       - prevent evaluation
  (and expr...)              - logical and
  (or expr...)               - logical or
  (when cond body...)        - if-then (no else)
  (unless cond body...)      - unless-then

Built-in Functions:
  + - * / mod neg abs max min sqrt
  < > <= >= = !=
  not
  print println input
  car cdr cons list len reverse nth concat range
  number? string? symbol? list? nil? function? type
  str string->number

Constants: true false nil
        """)

    def _print_env(self):
        for name, val in sorted(self.env.items()):
            if not name.startswith("_"):
                print(f"  {name} = {self._to_lisp_str(val)}")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        # Run a file
        filepath = sys.argv[1]
        with open(filepath, "r") as f:
            source = f.read()
        interp = Interpreter()
        interp.evaluate_and_print(source)
    else:
        # Interactive REPL
        interp = Interpreter()
        interp.repl()


if __name__ == "__main__":
    main()
