package matrix

import (
	"math/rand"
	"testing"
)

func TestAdd(t *testing.T) {
	scenarios := []struct {
		name      string
		a         [][]float64
		b         [][]float64
		expected  [][]float64
		expectErr bool
	}{
		{
			name:      "Mismatching Lengths",
			a:         [][]float64{{1, 2, 3}},
			b:         [][]float64{{1, 2, 3, 4, 5}},
			expected:  nil,
			expectErr: true,
		},
		{
			name:      "Mismatching Lengths Reversed",
			a:         [][]float64{{1, 2, 3, 4, 5, 6, 7}},
			b:         [][]float64{{1, 2, 3, 4, 5}},
			expected:  nil,
			expectErr: true,
		},
		{
			name:      "1x3",
			a:         [][]float64{{1, 2, 3}},
			b:         [][]float64{{1, 2, 3}},
			expected:  [][]float64{{2, 4, 6}},
			expectErr: false,
		},
		{
			name:      "1x8",
			a:         [][]float64{{1, 2, 3, 4, 5, 6, 7, 8}},
			b:         [][]float64{{1, 2, 3, 4, 5, 6, 7, 8}},
			expected:  [][]float64{{2, 4, 6, 8, 10, 12, 14, 16}},
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

			for i := range scenario.expected {
				for j := range scenario.expected[i] {
					if scenario.expected[i][j] != result[i][j] {
						t.Errorf("Add(%v, %v) expected %v but got %v at [%v, %v]", scenario.a, scenario.b, scenario.expected[i][j], result[i][j], i, j)
					}
				}
			}
		})
	}
}

func BenchmarkAdd(b *testing.B) {
	benchmark(b, Add)
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
		{
			name:      "3x2",
			a:         [][]float64{{1, 2}, {3, 4}, {5, 6}},
			b:         [][]float64{{1, 2, 3}, {4, 5, 6}},
			expected:  [][]float64{{9, 12, 15}, {19, 26, 33}, {29, 40, 51}},
			expectErr: false,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			result, err := Multiply(scenario.a, scenario.b)

			if scenario.expectErr && err == nil {
				t.Errorf("Multiply(%v, %v) expected error but got none", scenario.a, scenario.b)
			}

			if !scenario.expectErr && err != nil {
				t.Errorf("Multiply(%v, %v) expected no error but got one: %v", scenario.a, scenario.b, err)
			}

			for i, e := range scenario.expected {
				for j := range scenario.expected[i] {
					if e[j] != result[i][j] {
						t.Fatalf("Multiply(%v, %v) = %v, want %v", scenario.a, scenario.b, result, scenario.expected)
					}
				}
			}
		})
	}
}

func BenchmarkMultiply(b *testing.B) {
	benchmark(b, Multiply)
}

func TestTranspose(t *testing.T) {
	scenarios := []struct {
		name      string
		input     [][]float64
		expected  [][]float64
		expectErr bool
	}{
		{name: "nil", input: nil, expected: nil, expectErr: true},
		{name: "empty", input: [][]float64{}, expected: nil, expectErr: true},
		{name: "essentially empty", input: [][]float64{nil}, expected: nil, expectErr: true},
		{name: "nil entry", input: [][]float64{{1, 2, 3}, nil, {4, 5, 6}}, expected: nil, expectErr: true},
		{name: "imbalanced", input: [][]float64{{1, 2, 3}, {4, 5}, {6, 7, 8}}, expected: nil, expectErr: true},
		{name: "3x1", input: [][]float64{{1}, {4}, {7}}, expected: [][]float64{{1, 4, 7}}, expectErr: false},
		{name: "3x3", input: [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, expected: [][]float64{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}}, expectErr: false},
		{name: "3x4", input: [][]float64{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, expected: [][]float64{{1, 5, 9}, {2, 6, 10}, {3, 7, 11}, {4, 8, 12}}, expectErr: false},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			result, err := Transpose(scenario.input)

			if scenario.expectErr && err == nil {
				t.Fatal("Transpose expected error but got none")
			}

			if !scenario.expectErr && err != nil {
				t.Fatalf("Transpose expected no error but got one: %v", err)
			}

			if scenario.expectErr && err != nil {
				return
			}

			if result == nil {
				t.Fatalf("Transpose(%v) expected result but got nil", scenario.input)
			}

			for i := range scenario.expected {
				for j := range scenario.expected[i] {
					if scenario.expected[i][j] != result[i][j] {
						t.Fatalf("Transpose(%v) = %v, want %v at [%v, %v]", scenario.input, result[i][j], scenario.expected[i][j], i, j)
					}
				}
			}
		})
	}
}

func BenchmarkTranspose(b *testing.B) {
	l := 10000
	a := make([][]float64, l)

	for i := range a {
		a[i] = make([]float64, l)

		for j := range a[i] {
			a[i][j] = 1
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = Transpose(a)
	}
}

func benchmark(b *testing.B, benchmarkFn func(a, c [][]float64) ([][]float64, error)) {
	l := 1000
	a := make([][]float64, l)
	c := make([][]float64, l)

	for i := range l {
		a[i] = make([]float64, l)
		c[i] = make([]float64, l)

		a[i][i] = float64(rand.Intn(100))
		c[i][i] = float64(rand.Intn(100))
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = benchmarkFn(a, c)
	}
}
