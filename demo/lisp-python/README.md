# Simple Lisp Interpreter in Python

A complete Lisp interpreter implemented in Python (~900 lines), supporting arithmetic, logic, conditionals, loops, function definitions, recursion, and higher-order functions.

## Quick Start

```bash
# Run all demos with formatted output
python run_demo.py

# Run a .lisp file
python lisp.py demos.lisp

# Interactive REPL
python lisp.py
```

## Features

### Data Types
- **Numbers**: integers and floats (`42`, `3.14`)
- **Strings**: double-quoted (`"hello"`)
- **Symbols**: identifiers (`x`, `foo-bar`)
- **Lists**: S-expressions (`(1 2 3)`)
- **Booleans**: `true`, `false`
- **Nil**: `nil` (also represents empty)

### Special Forms
| Form | Description |
|------|-------------|
| `(def name value)` | Define a global variable |
| `(set! name value)` | Reassign a variable |
| `(defn name [params] body...)` | Define a function |
| `(fn [params] body...)` | Anonymous function |
| `(if cond then else)` | Conditional |
| `(cond [test result]...)` | Multi-way conditional (supports `else`) |
| `(let [(name val) ...] body...)` | Local bindings (parallel) |
| `(let* [(name val) ...] body...)` | Local bindings (sequential) |
| `(while condition body...)` | Loop while true |
| `(for var list body...)` | Iterate over a list |
| `(begin expr...)` | Sequential evaluation |
| `(quote expr)` / `'expr` | Prevent evaluation |
| `(and expr...)` | Short-circuit logical AND |
| `(or expr...)` | Short-circuit logical OR |
| `(when cond body...)` | If-then (no else) |
| `(unless cond body...)` | Unless-then |

### Built-in Functions

**Arithmetic**: `+` `-` `*` `/` `mod` `neg` `abs` `max` `min` `sqrt`

**Comparison**: `<` `>` `<=` `>=` `=` `!=`

**Logic**: `not`

**I/O**: `print` `println` `input`

**Lists**: `car` `cdr` `cons` `list` `len` `reverse` `nth` `concat` `range` `empty?`

**Types**: `number?` `string?` `symbol?` `list?` `nil?` `function?` `type`

**Strings**: `str` `string->number`

### Syntax
- Square brackets `[ ]` are aliases for parentheses (useful for parameter lists)
- Semicolon `;` starts a comment to end of line
- Single quote `'` prefix for quoting: `'(1 2 3)` → `(quote (1 2 3))`

## Examples

### Arithmetic
```lisp
(+ 1 2 3 4 5)        ; → 15
(sqrt 144)            ; → 12.0
(max 3 7 1 9 4)      ; → 9
```

### Functions
```lisp
(defn square [n]
    (* n n))

(square 7)            ; → 49

; Anonymous function
((fn [x] (* x 3)) 7)  ; → 21
```

### Recursion
```lisp
(defn factorial [n]
    (if (<= n 1)
        1
        (* n (factorial (- n 1)))))

(factorial 10)        ; → 3628800
```

### Loops
```lisp
; While loop
(def i 1)
(def total 0)
(while (<= i 10)
    (set! total (+ total i))
    (set! i (+ i 1)))

; For loop
(for item (list "a" "b" "c")
    (print item))
```

### Higher-Order Functions
```lisp
(defn my-map [f lst]
    (if (empty? lst)
        '()
        (cons (f (car lst))
              (my-map f (cdr lst)))))

(my-map square (list 1 2 3 4 5))  ; → (1 4 9 16 25)
```

### FizzBuzz
```lisp
(defn fizzbuzz [n]
    (cond
        ((and (= (mod n 3) 0) (= (mod n 5) 0)) "FizzBuzz")
        ((= (mod n 3) 0) "Fizz")
        ((= (mod n 5) 0) "Buzz")
        (else (str n))))
```

## Architecture

```
Source Code
    │
    ▼
┌─────────┐
│Tokenizer│  → flat token list
└─────────┘
    │
    ▼
┌────────┐
│ Reader │  → S-expressions (nested lists)
└────────┘
    │
    ▼
┌─────────┐
│ Evaluator │  → recursive evaluation with environment
└─────────┘
```

- **Tokenizer**: handles parens, strings, numbers, symbols, comments
- **Reader**: converts tokens to S-expressions (Lisp's AST)
- **Evaluator**: recursive descent with lexical scoping via environments

## Files

| File | Description |
|------|-------------|
| `lisp.py` | Interpreter implementation (~900 lines) |
| `run_demo.py` | Demo runner with formatted output |
| `demos.lisp` | Full demo script (all features) |

## REPL Commands

| Command | Description |
|---------|-------------|
| `.help` | Show help |
| `.env` | List environment variables |
| `.quit` / `.exit` | Exit REPL |
