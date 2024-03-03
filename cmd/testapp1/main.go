package main

import (
	"muratkoptur.com/gonn/v2/ad"
)

func main() {
	x1 := ad.NewVariable(2, nil, "")
	x2 := ad.NewVariable(5, nil, "")

	x1log := x1.Log()
	x1px2 := x1.Mul(x2)
	x2sin := x2.Sin()

	y := x1log.Add(x1px2).Sub(x2sin)
	y.Backward()
}
