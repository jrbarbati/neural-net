package matrix

import "testing"

func TestAdd(t *testing.T) {
	scenarios := []struct {
		name      string
		a         []float64
		b         []float64
		expected  []float64
		expectErr bool
	}{
		{
			name:      "Mismatching Lengths",
			a:         []float64{1, 2, 3},
			b:         []float64{1, 2, 3, 4, 5},
			expected:  nil,
			expectErr: true,
		},
		{
			name:      "Mismatching Lengths Reversed",
			a:         []float64{1, 2, 3, 4, 5, 6, 7},
			b:         []float64{1, 2, 3, 4, 5},
			expected:  nil,
			expectErr: true,
		},
		{
			name:      "1x3",
			a:         []float64{1, 2, 3},
			b:         []float64{1, 2, 3},
			expected:  []float64{2, 4, 6},
			expectErr: false,
		},
		{
			name:      "1x8",
			a:         []float64{1, 2, 3, 4, 5, 6, 7, 8},
			b:         []float64{1, 2, 3, 4, 5, 6, 7, 8},
			expected:  []float64{2, 4, 6, 8, 10, 12, 14, 16},
			expectErr: false,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			result, err := Add(scenario.a, scenario.b)

			if scenario.expectErr && err == nil {
				t.Errorf("Add(%v, %v) expected error but got none", scenario.a, scenario.b)
			}

			if !scenario.expectErr && err != nil {
				t.Errorf("Add(%v, %v) expected no error but got one: %v", scenario.a, scenario.b, err)
			}

			for i, r := range scenario.expected {
				if r != result[i] {
					t.Errorf("Add(%v, %v) = %v, want %v", scenario.a, scenario.b, result, scenario.expected)
				}
			}
		})
	}
}

func TestMultiply(t *testing.T) {
	scenarios := []struct {
		name      string
		a         [][]float64
		b         [][]float64
		expected  [][]float64
		expectErr bool
	}{
		{
			name:      "Mismatching Lengths",
			a:         [][]float64{{1}, {2}, {3}, {4}},
			b:         [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			expected:  nil,
			expectErr: true,
		},
		{
			name:      "Mismatching Lengths Reversed",
			a:         [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			b:         [][]float64{{1}, {2}, {3}, {4}},
			expected:  nil,
			expectErr: true,
		},
		{
			name:      "Happy Path 1",
			a:         [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			b:         [][]float64{{1}, {2}, {3}},
			expected:  [][]float64{{14}, {32}, {50}},
			expectErr: false,
		},
		{
			name:      "Happy Path 1 Reversed",
			a:         [][]float64{{1, 2, 3}},
			b:         [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			expected:  [][]float64{{30, 36, 42}},
			expectErr: false,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			result, err := Multiply(scenario.a, scenario.b)

			if scenario.expectErr && err == nil {
				t.Errorf("Add(%v, %v) expected error but got none", scenario.a, scenario.b)
			}

			if !scenario.expectErr && err != nil {
				t.Errorf("Add(%v, %v) expected no error but got one: %v", scenario.a, scenario.b, err)
			}

			for i, e := range scenario.expected {
				for j := range scenario.expected[i] {
					if e[j] != result[i][j] {
						t.Fatalf("Add(%v, %v) = %v, want %v", scenario.a, scenario.b, result, scenario.expected)
					}
				}
			}
		})
	}
}

func TestTranspose(t *testing.T) {
	scenarios := []struct {
		name     string
		input    [][]float64
		expected []float64
	}{
		{name: "Happy Path", input: [][]float64{{1}, {4}, {7}}, expected: []float64{1, 4, 7}},
		{name: "Happy Path w/ Extra", input: [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, expected: []float64{1, 4, 7}},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			result := Transpose(scenario.input)

			for i, e := range scenario.expected {
				if e != result[i] {
					t.Fatalf("Transpose(%v) = %v, want %v", scenario.input, result, scenario.expected)
				}
			}
		})
	}
}
