;; Simple Lisp example demonstrating arithmetic, conditionals, and function calls

;; Arithmetic: (1 + 2) * (3 - 4)
(+ (* 1 2) (- 3 4))

;; Conditional: if (a < b) then 10 else 20
a = 5
b = 10
(if (< a b) 10 20)

;; Nested function call with recursion simulation
(define (factorial n)
  (if (= n 0) 1 (* n (factorial (- n 1))))

;; Using the parser to evaluate expressions
(+ 1 2 3 4 5)
