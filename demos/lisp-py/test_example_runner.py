#!/usr/bin/env python3
"""Runner for test files using the lisp interpreter."""

from lisp import run_lisp

def main():
    # Test 1: test_example.lisp (after stripping comments)
    print("=== Testing test_example.lisp ===")
    try:
        result = run_lisp('''(+ (* 1 2) (- 3 4))
(a = 5
b = 10
(if (< a b) 10 20)
(define (factorial n)
  (if (= n 0) 1 (* n (factorial (- n 1))))
(+ 1 2 3 4 5)')')
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: demo1.lisp (after stripping comments)
    print("\n=== Testing demo1.lisp ===")
    try:
        result = run_lisp('''(+ 10 20)
(* 5 6)
(- 100 50)
(/ 200 40)
''')
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
