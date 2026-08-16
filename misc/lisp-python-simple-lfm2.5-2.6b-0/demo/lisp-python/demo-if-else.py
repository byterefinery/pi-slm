#!/usr/bin/env python3
"""Demo 3: Conditional expression (if)."""
# ((if true 1 2) 'a') -> 1
result = evaluate([['if', ['true', [1, None]], ['False', ['2', None]]]])
print("Result:", result)
