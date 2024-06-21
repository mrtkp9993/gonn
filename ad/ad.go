// this code is direct translation of my python implementation of autodiff
//
// https://github.com/mrtkp9993/autodiff

package ad

import (
	"fmt"
	"math"
)

var identifier int64 = 1

type Variable struct {
	f          float64
	d          float64
	op         string
	backwardfn func()
	parents    []*Variable
	name       string
}

func NewVariable(f float64, parents []*Variable, op string) *Variable {
	v := &Variable{
		f:          f,
		d:          0,
		op:         op,
		backwardfn: nil,
		parents:    parents,
		name:       fmt.Sprintf("v%d", identifier),
	}

	identifier++

	return v
}

func (v *Variable) GetData() float64 {
	return v.f
}

func (v *Variable) SetData(f float64) {
	v.f = f
}

func (v *Variable) GetGrad() float64 {
	return v.d
}

func (v *Variable) SetGrad(d float64) {
	v.d = d
}

func (v *Variable) GetOp() string {
	return v.op
}

func (v *Variable) SetOp(op string) {
	v.op = op
}

func (v *Variable) GetBackwardfn() func() {
	return v.backwardfn
}

func (v *Variable) SetBackwardfn(backwardfn func()) {
	v.backwardfn = backwardfn
}

func (v *Variable) GetName() string {
	return v.name
}

func (v *Variable) GetParents() []*Variable {
	return v.parents
}

func (v *Variable) String() string {
	return fmt.Sprintf("Variable(data=%f, grad=%f, op=%s, name=%s)", v.f, v.d, v.op, v.name)
}

func (v *Variable) Add(other *Variable) *Variable {
	out := NewVariable(v.f+other.f, []*Variable{v, other}, "+")
	out.backwardfn = func() {
		v.d += out.d
		other.d += out.d
	}
	return out
}

func (v *Variable) Sub(other *Variable) *Variable {
	out := NewVariable(v.f-other.f, []*Variable{v, other}, "-")
	out.backwardfn = func() {
		v.d += out.d
		other.d -= out.d
	}
	return out
}

func (v *Variable) Mul(other *Variable) *Variable {
	out := NewVariable(v.f*other.f, []*Variable{v, other}, "*")
	out.backwardfn = func() {
		v.d += other.f * out.d
		other.d += v.f * out.d
	}
	return out
}

func (v *Variable) Pow(other float64) *Variable {
	out := NewVariable(math.Pow(v.f, other), []*Variable{v}, fmt.Sprintf("**%f", other))
	out.backwardfn = func() {
		v.d += other * math.Pow(v.f, other-1) * out.d
	}
	return out
}

func (v *Variable) Div(other *Variable) *Variable {
	return v.Mul(other.Pow(-1))
}

func (v *Variable) Log() *Variable {
	out := NewVariable(math.Log(v.f), []*Variable{v}, "log")
	out.backwardfn = func() {
		v.d += (1 / v.f) * out.d
	}
	return out
}

func (v *Variable) Logb(base float64) *Variable {
	out := NewVariable(math.Log(v.f)/math.Log(base), []*Variable{v}, fmt.Sprintf("log_(%f)", base))
	out.backwardfn = func() {
		v.d += (1 / (v.f * math.Log(base))) * out.d
	}
	return out
}

func (v *Variable) Exp() *Variable {
	out := NewVariable(math.Exp(v.f), []*Variable{v}, "exp")
	out.backwardfn = func() {
		v.d += math.Exp(v.f) * out.d
	}
	return out
}

func (v *Variable) Root(other float64) *Variable {
	out := NewVariable(math.Pow(v.f, 1/other), []*Variable{v}, fmt.Sprintf("root(%f)", other))
	out.backwardfn = func() {
		v.d += (1 / other) * math.Pow(v.f, 1/other-1) * out.d
	}
	return out
}

func (v *Variable) Abs() *Variable {
	out := NewVariable(math.Abs(v.f), []*Variable{v}, "abs")
	out.backwardfn = func() {
		v.d += (math.Abs(v.f) / v.f) * out.d
	}
	return out
}

func (v *Variable) Sin() *Variable {
	out := NewVariable(math.Sin(v.f), []*Variable{v}, "sin")
	out.backwardfn = func() {
		v.d += math.Cos(v.f) * out.d
	}
	return out
}

func (v *Variable) Cos() *Variable {
	out := NewVariable(math.Cos(v.f), []*Variable{v}, "cos")
	out.backwardfn = func() {
		v.d += -math.Sin(v.f) * out.d
	}
	return out
}

func (v *Variable) Tan() *Variable {
	out := NewVariable(math.Tan(v.f), []*Variable{v}, "tan")
	out.backwardfn = func() {
		v.d += (1 / math.Pow(math.Cos(v.f), 2)) * out.d
	}
	return out
}

func (v *Variable) ASin() *Variable {
	out := NewVariable(math.Asin(v.f), []*Variable{v}, "asin")
	out.backwardfn = func() {
		v.d += (1 / math.Sqrt(1-math.Pow(v.f, 2))) * out.d
	}
	return out
}

func (v *Variable) ACos() *Variable {
	out := NewVariable(math.Acos(v.f), []*Variable{v}, "acos")
	out.backwardfn = func() {
		v.d += (-1 / math.Sqrt(1-math.Pow(v.f, 2))) * out.d
	}
	return out
}

func (v *Variable) ATan() *Variable {
	out := NewVariable(math.Atan(v.f), []*Variable{v}, "atan")
	out.backwardfn = func() {
		v.d += (1 / (1 + math.Pow(v.f, 2))) * out.d
	}
	return out
}

func (v *Variable) Sinh() *Variable {
	out := NewVariable(math.Sinh(v.f), []*Variable{v}, "sinh")
	out.backwardfn = func() {
		v.d += math.Cosh(v.f) * out.d
	}
	return out
}

func (v *Variable) Cosh() *Variable {
	out := NewVariable(math.Cosh(v.f), []*Variable{v}, "cosh")
	out.backwardfn = func() {
		v.d += math.Sinh(v.f) * out.d
	}
	return out
}

func (v *Variable) Tanh() *Variable {
	out := NewVariable(math.Tanh(v.f), []*Variable{v}, "tanh")
	out.backwardfn = func() {
		v.d += (1 - math.Pow(math.Tanh(v.f), 2)) * out.d
	}
	return out
}

func (v *Variable) ASinh() *Variable {
	out := NewVariable(math.Asinh(v.f), []*Variable{v}, "asinh")
	out.backwardfn = func() {
		v.d += (1 / math.Sqrt(math.Pow(v.f, 2)+1)) * out.d
	}
	return out
}

func (v *Variable) ACosh() *Variable {
	out := NewVariable(math.Acosh(v.f), []*Variable{v}, "acosh")
	out.backwardfn = func() {
		v.d += (1 / math.Sqrt(math.Pow(v.f, 2)-1)) * out.d
	}
	return out
}

func (v *Variable) ATanh() *Variable {
	out := NewVariable(math.Atanh(v.f), []*Variable{v}, "atanh")
	out.backwardfn = func() {
		v.d += (1 / (1 - math.Pow(v.f, 2))) * out.d
	}
	return out
}

func (v *Variable) Erf() *Variable {
	out := NewVariable(math.Erf(v.f), []*Variable{v}, "erf")
	out.backwardfn = func() {
		v.d += (2 / math.Sqrt(math.Pi)) * math.Exp(-math.Pow(v.f, 2)) * out.d
	}
	return out
}

func (v *Variable) Erfc() *Variable {
	out := NewVariable(math.Erfc(v.f), []*Variable{v}, "erfc")
	out.backwardfn = func() {
		v.d += (-2 / math.Sqrt(math.Pi)) * math.Exp(-math.Pow(v.f, 2)) * out.d
	}
	return out
}

func (v *Variable) ErfInv() *Variable {
	out := NewVariable(math.Erfinv(v.f), []*Variable{v}, "erfinv")
	out.backwardfn = func() {
		v.d += (2 / math.Sqrt(math.Pi)) * math.Exp(math.Pow(math.Erfinv(v.f), 2)) * out.d
	}
	return out
}

func (v *Variable) ErfcInv() *Variable {
	out := NewVariable(math.Erfcinv(v.f), []*Variable{v}, "erfcinv")
	out.backwardfn = func() {
		v.d += (-2 / math.Sqrt(math.Pi)) * math.Exp(math.Pow(math.Erfcinv(v.f), 2)) * out.d
	}
	return out
}

func buildTopo(v *Variable, topo []*Variable, visited map[*Variable]bool) []*Variable {
	if !visited[v] {
		visited[v] = true
		for _, child := range v.GetParents() {
			topo = buildTopo(child, topo, visited)
		}
		topo = append(topo, v)
	}
	return topo
}

func (v *Variable) Backward() {
	topo := buildTopo(v, []*Variable{}, map[*Variable]bool{})

	v.d = 1

	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backwardfn != nil {
			topo[i].backwardfn()
		}
	}
}

// ACTIVATIONS
// Tanh were already implemented above
func (x *Variable) Identity() *Variable {
	out := NewVariable(
		x.GetData(),
		[]*Variable{x},
		"Identity",
	)
	out.backwardfn = func() {
		x.SetGrad(x.GetGrad() + out.GetGrad())
	}
	return out
}

func (x *Variable) Sigmoid() *Variable {
	out := NewVariable(
		1/(1+math.Exp(-x.GetData())),
		[]*Variable{x},
		"Sigmoid",
	)
	out.backwardfn = func() {
		x.SetGrad(x.GetGrad() + out.GetGrad()*(1-out.GetData())*out.GetData())
	}
	return out
}

func (x *Variable) ReLU() *Variable {
	out := NewVariable(
		math.Max(0, x.GetData()),
		[]*Variable{x},
		"ReLU",
	)
	out.backwardfn = func() {
		if x.GetData() > 0 {
			x.SetGrad(x.GetGrad() + out.GetGrad())
		}
	}
	return out
}

func (x *Variable) LeakyReLU(alpha float64) *Variable {
	out := NewVariable(
		math.Max(0, x.GetData())+alpha*math.Min(0, x.GetData()),
		[]*Variable{x},
		"LeakyReLU",
	)
	out.backwardfn = func() {
		if x.GetData() > 0 {
			x.SetGrad(x.GetGrad() + out.GetGrad())
		} else {
			x.SetGrad(x.GetGrad() + alpha*out.GetGrad())
		}
	}
	return out
}
