package activation

import "math"

type Function struct {
	Apply      func([]float64) []float64
	Derivative func([]float64) []float64
}

var Sigmoid = Function{
	Apply: func(x []float64) []float64 {
		result := make([]float64, len(x))

		for i, num := range x {
			result[i] = 1 / (1 + math.Exp(-num))
		}

		return result
	},
	Derivative: func(x []float64) []float64 {
		result := make([]float64, len(x))

		for i, num := range x {
			result[i] = num * (1 - num)
		}

		return result
	},
}

var Tanh = Function{
	Apply: func(x []float64) []float64 {
		result := make([]float64, len(x))

		for i, num := range x {
			result[i] = (math.Exp(num) - math.Exp(-num)) / (math.Exp(num) + math.Exp(-num))
		}

		return result
	},
	Derivative: func(x []float64) []float64 {
		result := make([]float64, len(x))

		for i, num := range x {
			result[i] = 1 - num*num
		}

		return result
	},
}

var ReLU = Function{
	Apply: func(x []float64) []float64 {
		result := make([]float64, len(x))

		for i, num := range x {
			result[i] = math.Max(0, num)
		}

		return result
	},
	Derivative: func(x []float64) []float64 {
		result := make([]float64, len(x))

		for i, num := range x {
			if num > 0 {
				result[i] = 1
			} else {
				result[i] = 0
			}
		}

		return result
	},
}

var LeakyReLU = Function{
	Apply: func(x []float64) []float64 {
		result := make([]float64, len(x))

		for i, num := range x {
			if num > 0 {
				result[i] = num
			} else {
				result[i] = 0.01 * num
			}
		}

		return result
	},
	Derivative: func(x []float64) []float64 {
		result := make([]float64, len(x))

		for i, num := range x {
			if num > 0 {
				result[i] = 1
			} else {
				result[i] = 0.01
			}
		}

		return result
	},
}
