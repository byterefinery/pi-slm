#!/usr/bin/env python3
"""Test the lisp parser against arithmetic, logical operations, if statements, and loops."""

import sys
sys.path.insert(0, '/home/mtasic/projects-b/pi-slm/demos/lisp-py')

from lisp import Lexer, Parser, Evaluator

def test_arithmetic():
    """Test arithmetic operations (+, -, *, /) including nested calls."""
    print("=== Arithmetic Tests ===")
    
    # Simple arithmetic
    expr1 = "3 + 4"
    ast1 = Parser().parse(expr1)
    print(f"'{expr1}' -> {ast1}")
    
    # Nested arithmetic
    expr2 = "(+ 1 (* 2 3))"
    ast2 = Parser().parse(expr2)
    print(f"'{expr2}' -> {ast2}")
    
    # More complex nested
    expr3 = "((+ 1 (- 10 5)) * 2)"
    ast3 = Parser().parse(expr3)
    print(f"'{expr3}' -> {ast3}")
    
    # Division
    expr4 = "(/ 20 4)"
    ast4 = Parser().parse(expr4)
    print(f"'{expr4}' -> {ast4}")

def test_logical_operations():
    """Test logical operations (and, or, not)."""
    print("\n=== Logical Operations Tests ===")
    
    # And
    expr1 = "(and 1 2)"
    ast1 = Parser().parse(expr1)
    print(f"'{expr1}' -> {ast1}")
    
    # Or
    expr2 = "(or 0 5)"
    ast2 = Parser().parse(expr2)
    print(f"'{expr2}' -> {ast2}")
    
    # Not (unary)
    expr3 = "(not true)"
    ast3 = Parser().parse(expr3)
    print(f"'{expr3}' -> {ast3}")
    
    # Nested logical
    expr4 = "(and (or 1 2) true)"
    ast4 = Parser().parse(expr4)
    print(f"'{expr4}' -> {ast4}")

def test_if_statements():
    """Test if statements."""
    print("\n=== If Statements Tests ===")
    
    # Simple if
    expr1 = "(if true 1 2)"
    ast1 = Parser().parse(expr1)
    print(f"'{expr1}' -> {ast1}")
    
    # If with nested condition
    expr2 = "(if (and 1 2) 3 4)"
    ast2 = Parser().parse(expr2)
    print(f"'{expr2}' -> {ast2}")
    
    # If with else
    expr3 = "(if true 5 (else 10))"
    ast3 = Parser().parse(expr3)
    print(f"'{expr3}' -> {ast3}")

def test_loops():
    """Test loop constructs."""
    print("\n=== Loop Tests ===")
    
    # While loop
    expr1 = "(while true do 1)"
    ast1 = Parser().parse(expr1)
    print(f"'{expr1}' -> {ast1}")
    
    # For loop (if present in the parser)
    expr2 = "(for i from 0 to 3 do i)"
    ast2 = Parser().parse(expr2)
    print(f"'{expr2}' -> {ast2}")

def test_complex_expression():
    """Test a complex expression combining all features."""
    print("\n=== Complex Expression Test ===")
    
    expr = "(if (and (> 1 2) (< 5 10)) (+ (* 3 4) 2) (else (- 20 15)))"
    ast = Parser().parse(expr)
    print(f"'{expr}' -> {ast}")

if __name__ == "__main__":
    test_arithmetic()
    test_logical_operations()
    test_if_statements()
    test_loops()
    test_complex_expression()
