;; Demo 4: Loops - while
(defun count-down n)
  (while (> n 0)
    (print n)
    (- n 1)))
(count-down 3)
