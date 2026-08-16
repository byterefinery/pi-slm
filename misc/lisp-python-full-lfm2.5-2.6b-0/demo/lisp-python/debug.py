#!/usr/bin/env python3
"""Debug script to understand the parsing behavior."""

import ast

def parse_s_expr(code: str) -> Union[List[Any], Any]:
    """Parse a Lisp expression string into an AST."""
    code = code.strip()
    if not code:
        return None
    
    # If it starts with '(', it's a list (S-expression)
    if code.startswith('('):
        depth = 0
        i = 1
        while i < len(code) and depth > 0:
            if code[i] == '(':
                depth += 1
            elif code[i] == ')':
                depth -= 1
            i += 1
        
        inner = code[1:i-1]
        print(f"  Parsing list: inner='{inner}'")
        if not inner:
            print("  Inner is empty, returning None")
            return None
        result = parse_s_expr(inner)
        print(f"  Result for list: {result}")
        return [result]
    else:
        try:
            return float(code)
        except ValueError:
            return code

# Test various expressions
test_cases = [
    "(1)",
    "(+ 1 2)",
    "(* 10 5)",
    "(if (= 0 0) \"zero\" \"non-zero\")",
]

for tc in test_cases:
    print(f"Testing: {tc}")
    try:
        tree = parse_s_expr(tc)
        print(f"  Result: {tree}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
