package neuron

import "testing"

func TestNeuron_UpdateWeights(t *testing.T) {
	scenarios := []struct {
		name   string
		neuron *Neuron
		update []float64
	}{
		{
			name: "update weights",
			neuron: &Neuron{
				Weights: []float64{0.1, 0.2, 0.3},
			},
			update: []float64{0.01, 0.02, 0.03},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			scenario.neuron.UpdateWeights(scenario.update)

			if len(scenario.neuron.Weights) != len(scenario.update) {
				t.Errorf("expected %d weights, got %d", len(scenario.update), len(scenario.neuron.Weights))
			}

			for i := range scenario.neuron.Weights {
				if scenario.update[i] != scenario.neuron.Weights[i] {
					t.Errorf("expected weight %f, got %f", scenario.update[i], scenario.neuron.Weights[i])
				}
			}
		})
	}
}

func TestNeuron_UpdateBias(t *testing.T) {
	scenarios := []struct {
		name   string
		neuron *Neuron
		update float64
	}{
		{
			name: "update weights",
			neuron: &Neuron{
				Bias: 0.3,
			},
			update: 0.03,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			scenario.neuron.UpdateBias(scenario.update)

			if scenario.update != scenario.neuron.Bias {
				t.Errorf("expected weight %f, got %f", scenario.update, scenario.neuron.Bias)
			}
		})
	}
}
