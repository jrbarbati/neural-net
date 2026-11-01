package layer

import (
	"fmt"
	"math/rand"

	"github.com/jrbarbati/neural-net/internal/activation"
	"github.com/jrbarbati/neural-net/internal/matrix"
)

type Layer struct {
	weights      [][]float64
	biases       [][]float64
	activationFn activation.Function
}

type Config struct {
	InputLen     int
	Size         int
	ActivationFn activation.Function
}

func New(config Config) *Layer {
	w := make([][]float64, config.Size)
	b := make([][]float64, config.Size)

	for i := range w {
		w[i] = make([]float64, config.InputLen)
		b[i] = make([]float64, 1)

		for j := range w[i] {
			w[i][j] = rand.Float64()
		}
		b[i][0] = rand.Float64()
	}

	return &Layer{w, b, config.ActivationFn}
}

// ActivationValues takes the activation values from the previous layer and performs the necessary matrix math to output
// its own activation values
func (l *Layer) ActivationValues(activations [][]float64) ([][]float64, error) {
	result, err := matrix.Multiply(l.weights, activations)

	if err != nil {
		return nil, fmt.Errorf("error in matrix multiplication: %w", err)
	}

	activationValues, err := matrix.Add(result, l.biases)

	if err != nil {
		return nil, fmt.Errorf("error in matrix addition: %w", err)
	}

	return l.activationFn.Apply(activationValues), nil
}

// UpdateWeights calculates how this layer's weights should update based on the loss
func (l *Layer) UpdateWeights(loss []float64) {

}
