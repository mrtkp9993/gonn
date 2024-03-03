package ad

import "testing"

func TestFun1(t *testing.T) {
	x1 := NewVariable(2, nil, "")
	x2 := NewVariable(5, nil, "")

	x1log := x1.Log()
	x1px2 := x1.Mul(x2)
	x2sin := x2.Sin()

	y := x1log.Add(x1px2).Sub(x2sin)
	y.Backward()
	t.Log(y.String())
}
