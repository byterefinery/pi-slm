#!/usr/bin/env python3
"""Test script for lisp interpreter."""

from lisp-interpreter import evaluate, _eval_func

# Test 1: Simple addition
print("Test 1: + 5 3")
try:
    result = evaluate([['+', [5, 3]]])
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Simple subtraction
print("\nTest 2: - 10 4")
try:
    result = evaluate([['-', [10, 4]]])
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Multiplication
print("\nTest 3: * 6 7")
try:
    result = evaluate([['*', [6, 7]]])
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Division
print("\nTest 4: / 10 2")
try:
    result = evaluate([['/', [10, 2]]])
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Test 5: Logical and
print("\nTest 5: and true false")
try:
    result = evaluate([['and', [True, False]]])
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Test 6: Logical or
print("\nTest 6: or false true")
try:
    result = evaluate([['or', [False, True]]])
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Test 7: Not
print("\nTest 7: not true")
try:
    result = evaluate([['not', [True]]])
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
