package activation

import (
	"math"
)

type Function struct {
	Apply      func([][]float64) [][]float64
	Derivative func([][]float64) [][]float64
}

var Sigmoid = Function{
	Apply: func(x [][]float64) [][]float64 {
		result := make([][]float64, len(x))

		for i := range x {
			num := x[i][0]
			result[i] = []float64{1 / (1 + math.Exp(-num))}
		}

		return result
	},
	Derivative: func(x [][]float64) [][]float64 {
		result := make([][]float64, len(x))

		for i := range x {
			num := x[i][0]
			result[i] = []float64{num * (1 - num)}
		}

		return result
	},
}

var Tanh = Function{
	Apply: func(x [][]float64) [][]float64 {
		result := make([][]float64, len(x))

		for i := range x {
			num := x[i][0]
			result[i] = []float64{(math.Exp(num) - math.Exp(-num)) / (math.Exp(num) + math.Exp(-num))}
		}

		return result
	},
	Derivative: func(x [][]float64) [][]float64 {
		result := make([][]float64, len(x))

		for i := range x {
			num := x[i][0]
			result[i] = []float64{1 - num*num}
		}

		return result
	},
}

var ReLU = Function{
	Apply: func(x [][]float64) [][]float64 {
		result := make([][]float64, len(x))

		for i := range x {
			num := x[i][0]
			result[i] = []float64{math.Max(0, num)}
		}

		return result
	},
	Derivative: func(x [][]float64) [][]float64 {
		result := make([][]float64, len(x))

		for i := range x {
			num := x[i][0]
			if num > 0 {
				result[i] = []float64{1}
			} else {
				result[i] = []float64{0}
			}
		}

		return result
	},
}

var LeakyReLU = Function{
	Apply: func(x [][]float64) [][]float64 {
		result := make([][]float64, len(x))

		for i := range x {
			num := x[i][0]
			if num > 0 {
				result[i] = []float64{num}
			} else {
				result[i] = []float64{0.01 * num}
			}
		}

		return result
	},
	Derivative: func(x [][]float64) [][]float64 {
		result := make([][]float64, len(x))

		for i := range x {
			num := x[i][0]
			if num > 0 {
				result[i] = []float64{1}
			} else {
				result[i] = []float64{0.01}
			}
		}

		return result
	},
}
