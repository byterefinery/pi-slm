;; Simple arithmetic operations
(+ 1 2 3)
(* 10 5)
(- 15 7)
(/ 20 4)

;; Logical operations
(= 1 1)
(= 1 2)
(not (> 5 3))
(and (= 1 1) (< 5 10))
(or (= 1 1) (> 5 3))

;; Conditionals
(if (< 5 10) "small" "large")
(if (= 0 0) "zero" "non-zero")

;; Variables
a 10
b (+ a 5)
(c (* b 2))

;; Function definitions and calls
(define (add x y)
  (+ x y))
(add 3 4)

(define (factorial n)
  (if (<= n 1) n (* n (factorial (- n 1)))))  ; recursive

(factorial 5)
