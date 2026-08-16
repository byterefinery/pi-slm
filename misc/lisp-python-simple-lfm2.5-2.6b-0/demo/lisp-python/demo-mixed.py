#!/usr/bin/env python3
"""Demo 4: Mixed arithmetic and logical."""
# ((if (> 5 3) (+ 1 2) (- 10 3))) -> 6
result = evaluate([['if', ['>', [5, 3]], ['+', [1, 2]], ['-', [10, 3]]]])
print("Result:", result)
