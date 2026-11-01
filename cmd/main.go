package main

import (
	"fmt"

	"github.com/jrbarbati/neural-net/internal/activation"
	"github.com/jrbarbati/neural-net/internal/layer"
	"github.com/jrbarbati/neural-net/internal/network"
)

func main() {
	layers := []layer.Config{
		{InputLen: 784, Size: 16, ActivationFn: activation.ReLU},
		{InputLen: 16, Size: 16, ActivationFn: activation.ReLU},
		{InputLen: 16, Size: 10, ActivationFn: activation.Sigmoid},
	}

	n := network.New(layers)

	inputs := make([][]float64, 784)

	for i := range inputs {
		inputs[i] = []float64{0}
	}

	categorize, err := n.Categorize(inputs)

	if err != nil {
		fmt.Println(err.Error())
		return
	}

	fmt.Println(categorize)
}
