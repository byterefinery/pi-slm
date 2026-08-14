; ═══════════════════════════════════════════════════════════════════════════════
;  Simple Lisp Interpreter - Demos
;  Run with: python run_demo.py  or  python lisp.py demos.lisp
; ═══════════════════════════════════════════════════════════════════════════════

; ── Section 1: Arithmetic ────────────────────────────────────────────────────

(print "=== Arithmetic ===")
(print)

(print "Basic operations:")
(print (+ 2 3))
(print (- 10 3))
(print (* 4 5))
(print (/ 10 3))

(print "Chained operations:")
(print (+ 1 2 3 4 5))
(print (- 100 10 5 2))

(print "Math functions:")
(print (sqrt 144))
(print (abs -42))
(print (max 3 7 1 9 4))
(print (min 3 7 1 9 4))
(print (mod 17 5))

(print "Comparison and logic:")
(print (< 3 5))
(print (> 10 3))
(print (= 5 5))
(print (!= 5 3))
(print (and true true))
(print (and true false))
(print (or false true))
(print (not true))
(print)

; ── Section 2: Variables ─────────────────────────────────────────────────────

(print "=== Variables ===")
(print)

(def x 42)
(def greeting "Hello, Lisp!")
(def pi-val 3.14159)

(print "x = ")
(print x)
(print "greeting = ")
(print greeting)

(set! x 100)
(print "x after set! = ")
(print x)
(print)

; ── Section 3: Conditionals ──────────────────────────────────────────────────

(print "=== Conditionals ===")
(print)

; if / else
(if (> x 50)
    (print "x is greater than 50")
    (print "x is 50 or less"))

; cond (multi-way)
(def score 85)
(cond
    ((>= score 90) (print "Grade: A"))
    ((>= score 80) (print "Grade: B"))
    ((>= score 70) (print "Grade: C"))
    ((>= score 60) (print "Grade: D"))
    (else (print "Grade: F")))

; when / unless
(when (number? x)
    (print "x is a number: ")
    (print x))

(unless (nil? x)
    (print "x is not nil"))
(print)

; ── Section 4: Functions ─────────────────────────────────────────────────────

(print "=== Functions ===")
(print)

; Define a function
(defn square [n]
    (* n n))

(print "square(5) = ")
(print (square 5))

; Function with multiple parameters
(defn greet [name greeting]
    (print greeting)
    (print ", ")
    (print name)
    (print "!"))

(greet "World" "Hello")
(greet "Lisp" "Welcome to")

; Function with multiple body expressions (returns last)
(defn add-and-print [a b]
    (print "Adding ")
    (print a)
    (print " + ")
    (print b)
    (+ a b))

(def result (add-and-print 3 7))
(print "Result: ")
(print result)

; Higher-order: passing functions as values
(defn apply-twice [f x]
    (f (f x)))

(print "(apply-twice square 2) = ")
(print (apply-twice square 2))

; Anonymous functions
(print "Anonymous function: ((fn [x] (* x 3)) 7) = ")
(print ((fn [x] (* x 3)) 7))
(print)

; ── Section 5: Local Bindings (let) ──────────────────────────────────────────

(print "=== Local Bindings ===")
(print)

(let [(a 10)
      (b 20)]
    (print "a + b = ")
    (print (+ a b)))

; let* (sequential - each can reference previous)
(let* [(x 5)
       (y (* x 2))
       (z (+ x y))]
    (print "x=5, y=x*2, z=x+y  =>  z = ")
    (print z))

; let shadows outer scope
(print "Outer x = ")
(print x)
(let [(x 999)]
    (print "Inner x = ")
    (print x))
(print "Back to outer x = ")
(print x)
(print)

; ── Section 6: Loops ─────────────────────────────────────────────────────────

(print "=== Loops ===")
(print)

; While loop - countdown
(print "Countdown:")
(def counter 5)
(while (> counter 0)
    (print counter)
    (set! counter (- counter 1)))
(print "Liftoff!")

; While loop - sum
(print)
(print "Sum 1 to 10:")
(def i 1)
(def sum 0)
(while (<= i 10)
    (set! sum (+ sum i))
    (set! i (+ i 1)))
(print "Sum = ")
(print sum)

; For loop - iterate over a list
(print)
(print "Iterating over a list:")
(for item (list "apple" "banana" "cherry")
    (print "  - ")
    (print item))

; For loop with range
(print)
(print "Even numbers from range:")
(for n (range 10)
    (when (= (mod n 2) 0)
        (print n)))
(print)

; ── Section 7: Recursion ─────────────────────────────────────────────────────

(print "=== Recursion ===")
(print)

; Factorial
(defn factorial [n]
    (if (<= n 1)
        1
        (* n (factorial (- n 1)))))

(print "5! = ")
(print (factorial 5))
(print "10! = ")
(print (factorial 10))

; Fibonacci
(defn fibonacci [n]
    (cond
        ((= n 0) 0)
        ((= n 1) 1)
        (else (+ (fibonacci (- n 1))
                 (fibonacci (- n 2))))))

(print "Fibonacci sequence (first 10):")
(def fib-i 0)
(while (< fib-i 10)
    (print (fibonacci fib-i))
    (set! fib-i (+ fib-i 1)))
(print)

; ── Section 8: Lists ─────────────────────────────────────────────────────────

(print "=== Lists ===")
(print)

(def my-list (list 1 2 3 4 5))
(print "my-list: ")
(print my-list)
(print "length: ")
(print (len my-list))

(print "first element (car): ")
(print (car my-list))
(print "rest (cdr): ")
(print (cdr my-list))
(print "third element (nth 2): ")
(print (nth 2 my-list))

(print "reversed: ")
(print (reverse my-list))

(print "cons 0 onto front: ")
(print (cons 0 my-list))

(print "concat lists: ")
(print (concat (list 1 2) (list 3 4) (list 5)))

; Map-like with for
(print "Doubled list elements:")
(for n (list 1 2 3 4 5)
    (print (* n 2)))
(print)

; ── Section 9: Type System ───────────────────────────────────────────────────

(print "=== Type System ===")
(print)

(print "(type 42): ")
(print (type 42))
(print "(type 3.14): ")
(print (type 3.14))
(print "(type \"hello\"): ")
(print (type "hello"))
(print "(type nil): ")
(print (type nil))
(print "(type (list 1 2)): ")
(print (type (list 1 2)))

(print "(number? 42): ")
(print (number? 42))
(print "(string? \"hi\"): ")
(print (string? "hi"))
(print "(list? (list 1 2)): ")
(print (list? (list 1 2)))
(print "(nil? nil): ")
(print (nil? nil))
(print "(function? square): ")
(print (function? square))
(print)

; ── Section 10: Strings ──────────────────────────────────────────────────────

(print "=== Strings ===")
(print)

(print (str "Hello" " " "World"))
(print (str "The answer is " (string->number "42")))

(print "Mixed types with str:")
(print (str "number: " 42 ", list: " (list 1 2)))
(print)

; ── Section 11: Quoting ──────────────────────────────────────────────────────

(print "=== Quoting ===")
(print)

(print "Quoted list (not evaluated):")
(print '(1 2 3))
(print "Quoted symbol:")
(print 'x)
(print "Nested quote:")
(print '((a b) (c d)))
(print)

; ── Section 12: Complex Example - FizzBuzz ───────────────────────────────────

(print "=== FizzBuzz ===")
(print)

(defn fizzbuzz [n]
    (cond
        ((and (= (mod n 3) 0) (= (mod n 5) 0)) "FizzBuzz")
        ((= (mod n 3) 0) "Fizz")
        ((= (mod n 5) 0) "Buzz")
        (else (str n))))

(def fb-i 1)
(while (<= fb-i 20)
    (print (fizzbuzz fb-i))
    (set! fb-i (+ fb-i 1)))
(print)

; ── Section 13: Primes ───────────────────────────────────────────────────────

(print "=== Primes up to 30 ===")
(print)

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

(def p-i 2)
(while (<= p-i 30)
    (when (is-prime p-i)
        (print p-i))
    (set! p-i (+ p-i 1)))
(print)

; ── Section 14: Higher-Order Functions ────────────────────────────────────────

(print "=== Higher-Order Functions ===")
(print)

; map: apply function to each element
(defn my-map [f lst]
    (if (empty? lst)
        '()
        (cons (f (car lst))
              (my-map f (cdr lst)))))

(print "map square over (1 2 3 4 5):")
(print (my-map square (list 1 2 3 4 5)))

; filter: keep elements where predicate is true
(defn my-filter [pred lst]
    (cond
        ((empty? lst) '())
        ((pred (car lst))
            (cons (car lst) (my-filter pred (cdr lst))))
        (else (my-filter pred (cdr lst)))))

(print "filter even from (1 2 3 4 5 6 7 8):")
(print (my-filter (fn [x] (= (mod x 2) 0)) (list 1 2 3 4 5 6 7 8)))

; reduce: fold a list
(defn my-reduce [f initial lst]
    (if (empty? lst)
        initial
        (my-reduce f (f initial (car lst)) (cdr lst))))

(print "sum via reduce: ")
(print (my-reduce + 0 (list 1 2 3 4 5)))
(print "product via reduce: ")
(print (my-reduce * 1 (list 2 3 4 5)))

(print)
(print "=== All demos complete! ===")
