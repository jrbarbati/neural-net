package matrix

import (
	"errors"
)

// Add performs matrix addition on a 2d array. Can return an error if the dimensions(a) != dimensions(b)
// Computes value and updates input array a
func Add(a, b [][]float64) ([][]float64, error) {
	if a == nil || b == nil {
		return nil, errors.New("nil input")
	}

	if len(a) != len(b) {
		return nil, errors.New("dimension mismatch")
	}

	for i := range a {
		if a[i] == nil || b[i] == nil {
			return nil, errors.New("nil input")
		}

		if len(a[i]) != len(b[i]) {
			return nil, errors.New("dimension mismatch")
		}

		for j := range a[i] {
			a[i][j] += b[i][j]
		}
	}

	return a, nil
}

// Multiply performs matrix multiplication on the 2d arrays. Can return error if the dimensions cannot be multiplied.
// Computes value and returns entirely new array
func Multiply(a, b [][]float64) ([][]float64, error) {
	if a == nil || b == nil || len(a) == 0 || len(b) == 0 || len(a[0]) != len(b) {
		return nil, errors.New("invalid input")
	}

	result := make([][]float64, len(a))

	for i := range result {
		result[i] = make([]float64, len(b[0]))
	}

	for i := range a {
		for j := range b[0] {
			for k := range b {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}

	return result, nil
}

// Transpose produces an NxM array from an MxN array
func Transpose(a [][]float64) ([][]float64, error) {
	if a == nil || len(a) == 0 || a[0] == nil {
		return nil, errors.New("invalid input")
	}

	result := make([][]float64, len(a[0]))

	for i := range result {
		result[i] = make([]float64, len(a))
	}

	for i := range a {
		if a[i] == nil || len(a[i]) == 0 || len(a[i]) != len(a[0]) {
			return nil, errors.New("invalid input")
		}

		for j := range a[i] {
			result[j][i] = a[i][j]
		}
	}

	return result, nil
}
