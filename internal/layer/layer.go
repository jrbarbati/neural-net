package layer

import (
	"fmt"

	"github.com/jrbarbati/neural-net/internal/activation"
	"github.com/jrbarbati/neural-net/internal/matrix"
	"github.com/jrbarbati/neural-net/internal/neuron"
)

type Layer struct {
	neurons []*neuron.Neuron
}

type Config struct {
	InputLen     int
	Size         int
	ActivationFn activation.Function
}

func New(config Config) *Layer {
	return &Layer{
		neurons: build(config.Size, config.InputLen),
	}
}

func (l *Layer) ForwardPass(inputs []float64) ([]float64, error) {
	layerWeights := l.neuronWeights()
	layerBiases := l.neuronBiases()

	inputCol := make([][]float64, len(inputs))
	for i, v := range inputs {
		inputCol[i] = []float64{v}
	}

	weighted, err := matrix.Multiply(layerWeights, inputCol)

	if err != nil {
		return nil, fmt.Errorf("failed to update weights: %w", err)
	}

	z := matrix.Transpose(weighted)

	summed, err := matrix.Add(z, layerBiases)

	if err != nil {
		return nil, fmt.Errorf("failed to add bias: %w", err)
	}

	return summed, nil
}

func (l *Layer) BackPropagation(inputs []float64, loss float64) {
}

func (l *Layer) neuronWeights() [][]float64 {
	layerWeights := make([][]float64, len(l.neurons))

	for i := range l.neurons {
		layerWeights[i] = l.neurons[i].Weights
	}

	return layerWeights
}

func (l *Layer) neuronBiases() []float64 {
	layerBiases := make([]float64, len(l.neurons))

	for i := range l.neurons {
		layerBiases[i] = l.neurons[i].Bias
	}

	return layerBiases
}

func build(size, inputLen int) []*neuron.Neuron {
	neurons := make([]*neuron.Neuron, size)

	for i := range size {
		neurons[i] = neuron.New(inputLen)
	}

	return neurons
}
