package matrix

import (
	"errors"
)

func Add(a, b []float64) ([]float64, error) {
	if len(a) != len(b) {
		return nil, errors.New("matrix dimensions do not match")
	}

	summed := make([]float64, len(a))

	for i := range a {
		summed[i] = a[i] + b[i]
	}

	return summed, nil
}

func Multiply(a, b [][]float64) ([][]float64, error) {
	if len(a[0]) != len(b) {
		return nil, errors.New("matrix dimensions do not match")
	}

	var result [][]float64

	for i := range a {
		var row []float64

		for col := range b[0] {
			var sum float64

			for k := range b {
				sum += a[i][k] * b[k][col]
			}

			row = append(row, sum)
		}

		result = append(result, row)
	}

	return result, nil
}

func Transpose(a [][]float64) []float64 {
	transposed := make([]float64, len(a))

	for i := range a {
		transposed[i] = a[i][0]
	}

	return transposed
}
