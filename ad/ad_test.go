package ad

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

var tol = 1e-5

func TestFun1(t *testing.T) {
	x1val := 3.0
	x2val := 4.0

	x1 := NewVariable(x1val, nil, "")
	x2 := NewVariable(x2val, nil, "")

	trm1 := x1.Sub(x2)
	trm2 := x1.Mul(x2)
	trm3 := trm1.Div(trm2)

	trm3.Backward()

	x1DerExpected := 1 / math.Pow(x1val, 2)
	x2DerExpected := -1 / math.Pow(x2val, 2)

	assert.Equal(t, x1val, x1.f)
	assert.InDelta(t, x1DerExpected, x1.d, tol)

	assert.Equal(t, x2val, x2.f)
	assert.InDelta(t, x2DerExpected, x2.d, tol)
}

func TestFun2(t *testing.T) {
	x1val := 1.0
	x2val := 2.0

	x1 := NewVariable(x1val, nil, "")
	x2 := NewVariable(x2val, nil, "")

	trm1 := x1.Exp()
	trm2 := x2.Tan()
	trm3 := trm1.Mul(trm2)
	trm4 := x2.Log()
	trm5 := trm3.Sub(trm4)

	trm5.Backward()

	assert.Equal(t, x1val, x1.f)
	assert.InDelta(t, -5.93955, x1.d, tol)

	assert.Equal(t, x2val, x2.f)
	assert.InDelta(t, 15.19645, x2.d, tol)
}

func TestFun3(t *testing.T) {
	x1val := 3.0
	x2val := 8.0

	x1 := NewVariable(x1val, nil, "")
	x2 := NewVariable(x2val, nil, "")

	trm1 := x1.Sin()
	trm2 := x2.Logb(7)
	trm3 := trm1.Div(trm2)
	trm4 := x1.Root(2)
	trm5 := trm3.Add(trm4)

	trm5.Backward()

	assert.Equal(t, x1val, x1.f)
	assert.InDelta(t, -0.63775, x1.d, tol)

	assert.Equal(t, x2val, x2.f)
	assert.InDelta(t, -0.00794, x2.d, tol)
}

func TestFun4(t *testing.T) {
	x1val := 2.0
	x2val := 2.0

	x1 := NewVariable(x1val, nil, "")
	x2 := NewVariable(x2val, nil, "")

	trm1 := x1.Cos()
	trm2 := x2.Abs()
	trm3 := trm1.Mul(trm2)

	trm3.Backward()

	assert.Equal(t, x1val, x1.f)
	assert.InDelta(t, -1.81859, x1.d, tol)

	assert.Equal(t, x2val, x2.f)
	assert.InDelta(t, -0.41615, x2.d, tol)
}
