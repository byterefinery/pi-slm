#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from lisp-interpreter import run

# Test simple arithmetic
result = run("(+ 1 2)")
print(f"Result: {result}")
