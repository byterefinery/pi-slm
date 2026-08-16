#!/usr/bin/env python3
"""Demo 2: Logical operations."""
# ((and true (True)) 'False) -> True
result = evaluate([['and', ['true', ['True']], ['False']]])
print("Result:", result)
