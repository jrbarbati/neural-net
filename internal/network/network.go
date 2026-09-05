package network

import (
	"fmt"

	"github.com/jrbarbati/neural-net/internal/layer"
)

type Network struct {
	Layers []*layer.Layer
}

func New(layers []layer.Config) *Network {
	return &Network{
		Layers: build(layers),
	}
}

func (n *Network) Categorize(inputs []float64) ([]float64, error) {
	return n.forwardPass(inputs)
}

func (n *Network) Train(inputBatch [][]float64) error {
	return nil
}

func (n *Network) forwardPass(inputs []float64) ([]float64, error) {
	networkInput := inputs
	var err error

	for _, l := range n.Layers {
		networkInput, err = l.ForwardPass(networkInput)

		if err != nil {
			return nil, fmt.Errorf("error in forward pass: %w", err)
		}
	}

	return networkInput, nil
}

func build(layerConfigs []layer.Config) []*layer.Layer {
	var layers []*layer.Layer

	for _, layerConfig := range layerConfigs {
		layers = append(layers, layer.New(layerConfig))
	}

	return layers
}
