#!/usr/bin/env python3
"""Demo 1: Arithmetic operations."""
# ((+ 5 3) (* 2 nil)) -> 16
result = evaluate([['+', [5, 3]], ['*', [2, None]]]
print("Result:", result)
