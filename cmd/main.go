package main

import (
	"fmt"

	"github.com/jrbarbati/neural-net/internal/activation"
	"github.com/jrbarbati/neural-net/internal/layer"
	"github.com/jrbarbati/neural-net/internal/network"
)

func main() {
	layers := []layer.Config{
		{InputLen: 5, Size: 10, ActivationFn: activation.ReLU},
		{InputLen: 10, Size: 15, ActivationFn: activation.ReLU},
		{InputLen: 15, Size: 5, ActivationFn: activation.ReLU},
		{InputLen: 5, Size: 2, ActivationFn: activation.Sigmoid},
	}

	n := network.New(layers)

	categorize, err := n.Categorize([]float64{0, 0, 0, 0, 0})

	if err != nil {
		fmt.Println(err.Error())
		return
	}

	fmt.Println(categorize)
}
