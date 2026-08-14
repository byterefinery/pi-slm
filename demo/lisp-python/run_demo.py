#!/usr/bin/env python3
"""
Demo runner for the Simple Lisp Interpreter.
Run this to see all Lisp features in action.
"""

import sys
import os

# Add the lisp-python directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lisp import Interpreter, read, LispError


def run_demo_section(interp, title, code):
    """Run a single demo section with a header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print()
    # Print the code (commented)
    for line in code.strip().split('\n'):
        stripped = line.strip()
        if stripped:
            print(f"  ; {stripped}")
    print()
    # Evaluate
    try:
        interp.evaluate(code)
    except LispError as e:
        print(f"  ⚠ Error: {e}")
    print()


def main():
    print("=" * 60)
    print("  Simple Lisp Interpreter - Feature Demos")
    print("=" * 60)
    print()
    print("This demo showcases:")
    print("  • Arithmetic operations")
    print("  • Variables and assignment")
    print("  • Conditionals (if, cond, when, unless)")
    print("  • Function definitions and calls")
    print("  • Loops (while, for)")
    print("  • Recursion")
    print("  • Lists and list operations")
    print("  • Higher-order functions")
    print()

    interp = Interpreter()

    # ── Arithmetic ────────────────────────────────────────────────────────
    run_demo_section(interp, "1. Arithmetic Operations", """
(print "Basic: (+ 2 3) = ")
(print (+ 2 3))
(print "Chained: (+ 1 2 3 4 5) = ")
(print (+ 1 2 3 4 5))
(print "Math: (sqrt 144) = ")
(print (sqrt 144))
(print "Compare: (< 3 5) = ")
(print (< 3 5))
(print "Logic: (and true (not false)) = ")
(print (and true (not false)))
""")

    # ── Variables ─────────────────────────────────────────────────────────
    run_demo_section(interp, "2. Variables", """
(def x 42)
(print "def x 42 => x = ")
(print x)
(set! x 100)
(print "set! x 100 => x = ")
(print x)
(def greeting "Hello, Lisp!")
(print "String variable: ")
(print greeting)
""")

    # ── Conditionals ──────────────────────────────────────────────────────
    run_demo_section(interp, "3. Conditionals", """
(def score 85)
(if (> score 90)
    (print "Grade: A")
    (print "Grade: B or below"))

(cond
    ((>= score 90) (print "A"))
    ((>= score 80) (print "B"))
    ((>= score 70) (print "C"))
    (else (print "D or F")))

(when (number? score)
    (print "score is a number: ")
    (print score))

(unless (nil? score)
    (print "score is not nil"))
""")

    # ── Functions ─────────────────────────────────────────────────────────
    run_demo_section(interp, "4. Function Definitions & Calls", """
(defn square [n]
    (* n n))

(print "square(7) = ")
(print (square 7))

(defn greet [name]
    (print "Hello, ")
    (print name)
    (print "!"))

(greet "World")

(defn add-then-print [a b]
    (print "Computing ")
    (print a)
    (print " + ")
    (print b)
    (+ a b))

(def result (add-then-print 3 7))
(print "Result = ")
(print result)

(print "Anonymous: ((fn [x] (* x 3)) 7) = ")
(print ((fn [x] (* x 3)) 7))
""")

    # ── Local Bindings ────────────────────────────────────────────────────
    run_demo_section(interp, "5. Local Bindings (let)", """
(let [(a 10)
      (b 20)]
    (print "let a=10, b=20: a+b = ")
    (print (+ a b)))

(let* [(x 5)
       (y (* x 2))
       (z (+ x y))]
    (print "let* x=5, y=x*2, z=x+y: z = ")
    (print z))
""")

    # ── Loops ─────────────────────────────────────────────────────────────
    run_demo_section(interp, "6. Loops", """
(print "Countdown:")
(def counter 5)
(while (> counter 0)
    (print counter)
    (set! counter (- counter 1)))
(print "Go!")

(print)
(print "List iteration:")
(for item (list "apple" "banana" "cherry")
    (print "  - ")
    (print item))

(print)
(print "Range loop:")
(for n (range 5)
    (print n))
""")

    # ── Recursion ─────────────────────────────────────────────────────────
    run_demo_section(interp, "7. Recursion", """
(defn factorial [n]
    (if (<= n 1)
        1
        (* n (factorial (- n 1)))))

(print "3! = ")
(print (factorial 3))
(print "5! = ")
(print (factorial 5))
(print "10! = ")
(print (factorial 10))

(defn fibonacci [n]
    (cond
        ((= n 0) 0)
        ((= n 1) 1)
        (else (+ (fibonacci (- n 1))
                 (fibonacci (- n 2))))))

(print)
(print "Fibonacci(10) = ")
(print (fibonacci 10))
""")

    # ── Lists ─────────────────────────────────────────────────────────────
    run_demo_section(interp, "8. Lists", """
(def nums (list 10 20 30 40 50))
(print "list: ")
(print nums)
(print "length: ")
(print (len nums))
(print "first (car): ")
(print (car nums))
(print "rest (cdr): ")
(print (cdr nums))
(print "third (nth 2): ")
(print (nth 2 nums))
(print "reversed: ")
(print (reverse nums))
(print "cons 0: ")
(print (cons 0 nums))
""")

    # ── Type System ───────────────────────────────────────────────────────
    run_demo_section(interp, "9. Type System", """
(print "(type 42) = ")
(print (type 42))
(print "(type nil) = ")
(print (type nil))
(print "(number? 42) = ")
(print (number? 42))
(print "(list? (list 1 2)) = ")
(print (list? (list 1 2)))
(print "(nil? nil) = ")
(print (nil? nil))
""")

    # ── Strings ───────────────────────────────────────────────────────────
    run_demo_section(interp, "10. Strings", """
(print (str "Hello" " " "World"))
(print (str "The number is " 42))
(print (str "number->str: " (string->number "123")))
""")

    # ── Quoting ───────────────────────────────────────────────────────────
    run_demo_section(interp, "11. Quoting", """
(print "Quoted list (literal, not evaluated):")
(print '(1 2 3))
(print "Nested:")
(print '((a b) (c d)))
""")

    # ── FizzBuzz ──────────────────────────────────────────────────────────
    run_demo_section(interp, "12. FizzBuzz (1-20)", """
(defn fizzbuzz [n]
    (cond
        ((and (= (mod n 3) 0) (= (mod n 5) 0)) "FizzBuzz")
        ((= (mod n 3) 0) "Fizz")
        ((= (mod n 5) 0) "Buzz")
        (else (str n))))

(def i 1)
(while (<= i 20)
    (print (fizzbuzz i))
    (set! i (+ i 1)))
""")

    # ── Primes ────────────────────────────────────────────────────────────
    run_demo_section(interp, "13. Primes up to 30", """
(defn is-prime [n]
    (cond
        ((< n 2) false)
        ((= n 2) true)
        ((= (mod n 2) 0) false)
        (else
            (let [(divisor 3)]
                (while (and (<= (* divisor divisor) n)
                            (!= (mod n divisor) 0))
                    (set! divisor (+ divisor 2)))
                (> (* divisor divisor) n)))))

(def p 2)
(while (<= p 30)
    (when (is-prime p)
        (print p))
    (set! p (+ p 1)))
""")

    # ── Higher-Order Functions ────────────────────────────────────────────
    run_demo_section(interp, "14. Higher-Order Functions", """
(defn my-map [f lst]
    (if (empty? lst)
        '()
        (cons (f (car lst))
              (my-map f (cdr lst)))))

(print "map square over (1 2 3 4 5):")
(print (my-map square (list 1 2 3 4 5)))

(defn my-filter [pred lst]
    (cond
        ((empty? lst) '())
        ((pred (car lst))
            (cons (car lst) (my-filter pred (cdr lst))))
        (else (my-filter pred (cdr lst)))))

(print "filter even from (1..8):")
(print (my-filter (fn [x] (= (mod x 2) 0)) (list 1 2 3 4 5 6 7 8)))

(defn my-reduce [f initial lst]
    (if (empty? lst)
        initial
        (my-reduce f (f initial (car lst)) (cdr lst))))

(print "sum via reduce: ")
(print (my-reduce + 0 (list 1 2 3 4 5)))
""")

    # ── Run full demo file if it exists ───────────────────────────────────
    demo_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos.lisp")
    if os.path.exists(demo_file):
        print("\n" + "=" * 60)
        print("  Running full demos.lisp file...")
        print("=" * 60)
        with open(demo_file) as f:
            source = f.read()
        interp2 = Interpreter()
        try:
            interp2.evaluate(source)
        except LispError as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("  All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
