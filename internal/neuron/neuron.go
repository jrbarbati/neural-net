package neuron

type Neuron struct {
	Weights []float64
	Bias    float64
}

func New(inputLen int) *Neuron {
	return &Neuron{
		Weights: make([]float64, inputLen),
		Bias:    0,
	}
}

func (n *Neuron) UpdateWeights(weights []float64) {
	n.Weights = weights
}

func (n *Neuron) UpdateBias(bias float64) {
	n.Bias = bias
}
