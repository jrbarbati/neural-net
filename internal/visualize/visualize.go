// Package visualize records what a network.Network does and renders it as an interactive page.
//
// Wrap a network in a Recorder and use the Recorder exactly as you would use the network. Every call is passed
// straight through to the real network (so results are identical), and the Recorder additionally captures:
//
//   - Categorize: the forward pass — the input, then z = W·a + b and a = f(z) for every layer.
//   - Train: the weights and biases before and after the call (so you can see the ΔW / Δb that backprop applied),
//     plus a forward trace of the input using the pre-update weights.
//
// If you do not wrap the network, nothing is captured and the network runs normally. Model is satisfied by both
// *network.Network and *Recorder, so code written against Model can take either:
//
//	var m visualize.Model = net                   // plain run
//	rec, _ := visualize.Wrap(net, visualize.Options{})
//	m = rec                                       // recorded run
//	go rec.ListenAndServe("localhost:8080")       // watch live in a browser, or…
//	defer rec.SaveHTML("network.html")            // …save a self-contained replay
//
// The page shows the input, hidden and output layers. Click a neuron to see its bias, z, activation and every
// incoming weight (with the input it carried and how much Train changed it). A timeline steps through the recorded
// calls; forward passes animate input → output and training updates animate output → input.
//
// Layer indexing used throughout this package: layer 0 is the input layer and layer i (i >= 1) is network.Layers[i-1].
//
// layer.Layer keeps its weights, biases and activation function in unexported fields, so they are read via
// reflection. If those fields are renamed or change type, Wrap/Capture return a descriptive error rather than panic.
//
// LLM-written
package visualize

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/jrbarbati/neural-net/internal/activation"
	"github.com/jrbarbati/neural-net/internal/network"
)

// Model is the part of the network API that the Recorder mirrors. Both *network.Network and *Recorder implement it.
type Model interface {
	Categorize(inputs [][]float64) ([][]float64, error)
	Train(inputBatch [][]float64) error
}

var (
	_ Model = (*network.Network)(nil)
	_ Model = (*Recorder)(nil)
)

// ---------------------------------------------------------------------------------------------------------------------
// Snapshots (reading parameters out of the network)
// ---------------------------------------------------------------------------------------------------------------------

// Snapshot is a copy of a network's parameters at a point in time.
type Snapshot struct {
	// InputSize is the number of neurons in the input layer (layer 0).
	InputSize int `json:"-"`
	// Layers holds every non-input layer, in order. Layers[0] is visualize layer 1.
	Layers []LayerSnapshot `json:"layers"`
}

// LayerSnapshot is a copy of one layer's parameters.
type LayerSnapshot struct {
	// Activation is the name of the layer's activation function ("ReLU", "Sigmoid", ...) or "custom".
	Activation string `json:"-"`
	// Weights[i][j] is the weight on the connection from neuron j of the previous layer into neuron i of this layer.
	Weights [][]num `json:"weights"`
	// Biases[i] is the bias of neuron i of this layer.
	Biases []num `json:"biases"`
}

// num is a float64 that survives JSON encoding even when it is NaN or ±Inf.
// It is written with float32 precision to keep recordings small.
type num float64

func (n num) MarshalJSON() ([]byte, error) {
	f := float64(n)
	switch {
	case math.IsNaN(f):
		return []byte(`"NaN"`), nil
	case math.IsInf(f, 1):
		return []byte(`"Infinity"`), nil
	case math.IsInf(f, -1):
		return []byte(`"-Infinity"`), nil
	case math.Abs(f) > math.MaxFloat32:
		return strconv.AppendFloat(nil, f, 'g', -1, 64), nil
	}
	return strconv.AppendFloat(nil, f, 'g', -1, 32), nil
}

// Size returns the number of neurons in the layer.
func (l LayerSnapshot) Size() int { return len(l.Biases) }

// InputLen returns the number of incoming connections per neuron.
func (l LayerSnapshot) InputLen() int {
	if len(l.Weights) == 0 {
		return 0
	}
	return len(l.Weights[0])
}

// Neuron returns the bias and incoming weights of a neuron. layerIdx follows the package convention
// (0 = input layer, which has no bias or incoming weights and therefore returns an error).
func (s *Snapshot) Neuron(layerIdx, neuronIdx int) (bias float64, weights []float64, err error) {
	if layerIdx == 0 {
		return 0, nil, errors.New("layer 0 is the input layer: its neurons have no bias or incoming weights")
	}
	if layerIdx < 0 || layerIdx > len(s.Layers) {
		return 0, nil, fmt.Errorf("layer %d out of range [0, %d]", layerIdx, len(s.Layers))
	}

	l := s.Layers[layerIdx-1]

	if neuronIdx < 0 || neuronIdx >= l.Size() {
		return 0, nil, fmt.Errorf("neuron %d out of range for layer %d (size %d)", neuronIdx, layerIdx, l.Size())
	}

	weights = make([]float64, len(l.Weights[neuronIdx]))
	for j, w := range l.Weights[neuronIdx] {
		weights[j] = float64(w)
	}

	return float64(l.Biases[neuronIdx]), weights, nil
}

// LayerName returns a human-readable name for a layer index ("Input", "Hidden 1", "Output", ...).
func (s *Snapshot) LayerName(layerIdx int) string {
	switch {
	case layerIdx == 0:
		return "Input"
	case layerIdx == len(s.Layers):
		return "Output"
	default:
		return fmt.Sprintf("Hidden %d", layerIdx)
	}
}

func (s *Snapshot) equal(o *Snapshot) bool {
	if len(s.Layers) != len(o.Layers) {
		return false
	}
	for k := range s.Layers {
		a, b := s.Layers[k], o.Layers[k]
		if len(a.Biases) != len(b.Biases) || len(a.Weights) != len(b.Weights) {
			return false
		}
		for i := range a.Biases {
			if math.Float64bits(float64(a.Biases[i])) != math.Float64bits(float64(b.Biases[i])) {
				return false
			}
		}
		for i := range a.Weights {
			if len(a.Weights[i]) != len(b.Weights[i]) {
				return false
			}
			for j := range a.Weights[i] {
				if math.Float64bits(float64(a.Weights[i][j])) != math.Float64bits(float64(b.Weights[i][j])) {
					return false
				}
			}
		}
	}
	return true
}

// Capture copies the weights, biases and activation names out of n.
func Capture(n *network.Network) (*Snapshot, error) {
	if n == nil {
		return nil, errors.New("nil network")
	}
	if len(n.Layers) == 0 {
		return nil, errors.New("network has no layers")
	}

	snap := &Snapshot{}

	for i, l := range n.Layers {
		if l == nil {
			return nil, fmt.Errorf("layer %d is nil", i+1)
		}

		ls, err := captureLayer(reflect.ValueOf(l).Elem())
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i+1, err)
		}

		if i == 0 {
			snap.InputSize = ls.InputLen()
		} else if prev := snap.Layers[i-1]; ls.InputLen() != prev.Size() {
			return nil, fmt.Errorf("layer %d expects %d inputs but layer %d has %d neurons", i+1, ls.InputLen(), i, prev.Size())
		}

		snap.Layers = append(snap.Layers, ls)
	}

	return snap, nil
}

func captureLayer(v reflect.Value) (LayerSnapshot, error) {
	weights, err := readMatrix(v, "weights")
	if err != nil {
		return LayerSnapshot{}, err
	}

	biasMatrix, err := readMatrix(v, "biases")
	if err != nil {
		return LayerSnapshot{}, err
	}

	if len(biasMatrix) != len(weights) {
		return LayerSnapshot{}, fmt.Errorf("%d bias rows but %d weight rows", len(biasMatrix), len(weights))
	}

	biases := make([]num, len(biasMatrix))
	for i, row := range biasMatrix {
		if len(row) != 1 {
			return LayerSnapshot{}, fmt.Errorf("bias row %d has %d columns, expected 1", i, len(row))
		}
		biases[i] = row[0]
	}

	for i, row := range weights {
		if len(row) != len(weights[0]) {
			return LayerSnapshot{}, fmt.Errorf("weight row %d has %d columns, expected %d", i, len(row), len(weights[0]))
		}
	}

	return LayerSnapshot{
		Activation: activationName(v.FieldByName("activationFn")),
		Weights:    weights,
		Biases:     biases,
	}, nil
}

// readMatrix reads a [][]float64 field (exported or not) from a struct value.
func readMatrix(v reflect.Value, field string) ([][]num, error) {
	f := v.FieldByName(field)

	if !f.IsValid() {
		return nil, fmt.Errorf("layer.Layer has no field %q (did the struct change?)", field)
	}

	if f.Type() != reflect.TypeOf([][]float64(nil)) {
		return nil, fmt.Errorf("layer.Layer field %q is %s, expected [][]float64", field, f.Type())
	}

	out := make([][]num, f.Len())

	for i := range out {
		row := f.Index(i)
		out[i] = make([]num, row.Len())

		for j := range out[i] {
			out[i][j] = num(row.Index(j).Float())
		}
	}

	return out, nil
}

// activationName identifies one of the activation package's predefined functions by comparing function pointers.
func activationName(fn reflect.Value) string {
	if !fn.IsValid() || fn.Type() != reflect.TypeOf(activation.Function{}) {
		return "unknown"
	}

	apply := fn.FieldByName("Apply")
	if apply.IsNil() {
		return "none"
	}

	known := []struct {
		name string
		fn   activation.Function
	}{
		{"ReLU", activation.ReLU},
		{"LeakyReLU", activation.LeakyReLU},
		{"Sigmoid", activation.Sigmoid},
		{"Tanh", activation.Tanh},
	}

	for _, k := range known {
		if reflect.ValueOf(k.fn.Apply).Pointer() == apply.Pointer() {
			return k.name
		}
	}

	return "custom"
}

// ---------------------------------------------------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------------------------------------------------

// Options controls how much a Recorder keeps.
type Options struct {
	// SampleEvery records 1 out of every N calls (counted separately for Categorize and Train). Unrecorded calls go
	// straight to the network with no overhead. Default 1 (record everything).
	SampleEvery int
	// MaxFrames is how many recorded calls to keep; older ones are dropped. Default 200.
	MaxFrames int
}

// Recorder wraps a network, passing every call through unchanged while recording what happened.
// Recording calls and serving/saving the visualization may happen concurrently.
type Recorder struct {
	net  *network.Network
	opts Options

	mu       sync.Mutex
	meta     metaJSON
	frames   []*frame
	seq      int
	calls    map[string]int
	last     *snapRec
	nextSnap int
	err      error
}

type snapRec struct {
	id   int
	snap *Snapshot
}

type metaJSON struct {
	InputSize int             `json:"inputSize"`
	Layers    []layerMetaJSON `json:"layers"`
}

type layerMetaJSON struct {
	Size       int    `json:"size"`
	Activation string `json:"activation"`
}

type frame struct {
	Seq   int     `json:"seq"`
	Kind  string  `json:"kind"` // "snapshot", "forward" or "train"
	Label string  `json:"label"`
	Snap  int     `json:"snap"`            // weights in effect for the call (before the update, for "train")
	After *int    `json:"after,omitempty"` // weights after Train returned
	Input []num   `json:"input,omitempty"`
	Z     [][]num `json:"z,omitempty"` // Z[k] = pre-activations of layer k+1
	A     [][]num `json:"a,omitempty"` // A[k] = activations of layer k+1
	Err   string  `json:"err,omitempty"`

	before, after *snapRec
}

// Wrap starts recording n. The returned Recorder records an initial snapshot of the weights immediately.
func Wrap(n *network.Network, opts Options) (*Recorder, error) {
	if opts.SampleEvery <= 0 {
		opts.SampleEvery = 1
	}
	if opts.MaxFrames <= 0 {
		opts.MaxFrames = 200
	}

	snap, err := Capture(n)
	if err != nil {
		return nil, err
	}

	r := &Recorder{net: n, opts: opts, calls: map[string]int{}}
	r.meta.InputSize = snap.InputSize
	for _, l := range snap.Layers {
		r.meta.Layers = append(r.meta.Layers, layerMetaJSON{Size: l.Size(), Activation: l.Activation})
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	rec := r.intern(snap)
	r.push(&frame{Kind: "snapshot", Label: "Initial weights", before: rec})

	return r, nil
}

// Network returns the wrapped network.
func (r *Recorder) Network() *network.Network { return r.net }

// Err returns the most recent error encountered while recording (not errors returned by the network itself).
// When recording fails, calls still go through to the network; they just aren't captured.
func (r *Recorder) Err() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.err
}

// Categorize runs n.Categorize and records the forward pass.
func (r *Recorder) Categorize(inputs [][]float64) ([][]float64, error) {
	idx, ok := r.sample("forward")
	if !ok {
		return r.net.Categorize(inputs)
	}

	before, ok := r.snapshot()
	if !ok {
		return r.net.Categorize(inputs)
	}

	f := &frame{Kind: "forward", Label: fmt.Sprintf("Categorize #%d", idx), before: before}
	f.Input, f.Z, f.A, f.Err = traceForward(r.net, before.snap, inputs)

	out, err := r.net.Categorize(inputs)
	if err != nil {
		f.Err = err.Error()
	}

	r.mu.Lock()
	r.push(f)
	r.mu.Unlock()

	return out, err
}

// Train runs n.Train and records the weights and biases before and after, so the update it applied can be seen.
// If inputBatch has the shape of a single input column vector, the forward pass over it is traced as well.
func (r *Recorder) Train(inputBatch [][]float64) error {
	idx, ok := r.sample("train")
	if !ok {
		return r.net.Train(inputBatch)
	}

	before, ok := r.snapshot()
	if !ok {
		return r.net.Train(inputBatch)
	}

	f := &frame{Kind: "train", Label: fmt.Sprintf("Train step %d", idx), before: before}
	if len(inputBatch) == before.snap.InputSize {
		f.Input, f.Z, f.A, f.Err = traceForward(r.net, before.snap, inputBatch)
	}

	err := r.net.Train(inputBatch)
	if err != nil {
		f.Err = err.Error()
	}

	if after, ok := r.snapshot(); ok {
		f.after = after
	}

	r.mu.Lock()
	r.push(f)
	r.mu.Unlock()

	return err
}

// sample reports whether this call should be recorded, and the 1-based call number for its kind.
func (r *Recorder) sample(kind string) (int, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls[kind]++
	n := r.calls[kind]
	return n, (n-1)%r.opts.SampleEvery == 0
}

// snapshot captures the current weights, reusing the previous snapshot when nothing has changed.
func (r *Recorder) snapshot() (*snapRec, bool) {
	snap, err := Capture(r.net)

	r.mu.Lock()
	defer r.mu.Unlock()

	if err != nil {
		r.err = fmt.Errorf("capturing weights: %w", err)
		return nil, false
	}

	return r.intern(snap), true
}

func (r *Recorder) intern(snap *Snapshot) *snapRec {
	if r.last != nil && r.last.snap.equal(snap) {
		return r.last
	}
	r.last = &snapRec{id: r.nextSnap, snap: snap}
	r.nextSnap++
	return r.last
}

func (r *Recorder) push(f *frame) {
	f.Seq = r.seq
	r.seq++
	f.Snap = f.before.id
	if f.after != nil {
		id := f.after.id
		f.After = &id
	}
	r.frames = append(r.frames, f)
	if over := len(r.frames) - r.opts.MaxFrames; over > 0 {
		r.frames = append([]*frame(nil), r.frames[over:]...)
	}
}

// traceForward reproduces the forward pass layer by layer using each layer's own ActivationValues, recording z and a.
// Only the first column of each matrix is recorded, matching how the activation functions treat their input.
func traceForward(n *network.Network, snap *Snapshot, inputs [][]float64) (in []num, z, a [][]num, errMsg string) {
	in = column0(inputs)
	cur := inputs

	for k, l := range n.Layers {
		ls := snap.Layers[k]

		if len(cur) != ls.InputLen() {
			return in, z, a, fmt.Sprintf("layer %d expects %d inputs, got %d", k+1, ls.InputLen(), len(cur))
		}

		zk := make([]num, ls.Size())
		for i := range zk {
			s := float64(ls.Biases[i])
			for j, w := range ls.Weights[i] {
				if len(cur[j]) > 0 {
					s += float64(w) * cur[j][0]
				}
			}
			zk[i] = num(s)
		}

		next, err := l.ActivationValues(cur)
		if err != nil {
			return in, z, a, fmt.Sprintf("layer %d: %v", k+1, err)
		}

		z = append(z, zk)
		a = append(a, column0(next))
		cur = next
	}

	return in, z, a, ""
}

func column0(m [][]float64) []num {
	out := make([]num, len(m))
	for i, row := range m {
		if len(row) == 0 {
			out[i] = num(math.NaN())
		} else {
			out[i] = num(row[0])
		}
	}
	return out
}

type payloadJSON struct {
	Meta      metaJSON          `json:"meta"`
	MaxFrames int               `json:"maxFrames"`
	Snapshots map[int]*Snapshot `json:"snapshots"`
	Frames    []*frame          `json:"frames"`
}

// payload encodes every frame newer than sinceSeq and every snapshot they reference that is newer than sinceSnap.
func (r *Recorder) payload(sinceSeq, sinceSnap int) ([]byte, error) {
	r.mu.Lock()
	p := payloadJSON{Meta: r.meta, MaxFrames: r.opts.MaxFrames, Snapshots: map[int]*Snapshot{}, Frames: []*frame{}}
	for _, f := range r.frames {
		if f.Seq <= sinceSeq {
			continue
		}
		p.Frames = append(p.Frames, f)
		for _, s := range []*snapRec{f.before, f.after} {
			if s != nil && s.id > sinceSnap {
				p.Snapshots[s.id] = s.snap
			}
		}
	}
	r.mu.Unlock()

	// Frames and snapshots are immutable once recorded, so encoding outside the lock is safe.
	return json.Marshal(p)
}

// ---------------------------------------------------------------------------------------------------------------------
// HTML output
// ---------------------------------------------------------------------------------------------------------------------

// WriteHTML writes a self-contained HTML page replaying everything recorded so far.
func (r *Recorder) WriteHTML(w io.Writer) error {
	data, err := r.payload(-1, -1)
	if err != nil {
		return fmt.Errorf("encoding recording: %w", err)
	}

	// json.Marshal escapes <, > and &, so the payload is safe to embed inside a <script> element.
	_, err = io.WriteString(w, strings.Replace(htmlTemplate, "/*__DATA__*/null", string(data), 1))
	return err
}

// SaveHTML writes the replay page to path (e.g. "network.html"). Open the file in any browser.
func (r *Recorder) SaveHTML(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating %s: %w", path, err)
	}

	if err := r.WriteHTML(f); err != nil {
		_ = f.Close()
		return err
	}

	return f.Close()
}

// Handler serves a live view: "/" is the page, which polls "/data" for new frames as they are recorded.
func (r *Recorder) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/" {
			http.NotFound(w, req)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		// No embedded data (null) → the page runs in live mode and polls /data.
		_, _ = io.WriteString(w, strings.Replace(htmlTemplate, "/*__DATA__*/null", "null", 1))
	})

	mux.HandleFunc("/data", func(w http.ResponseWriter, req *http.Request) {
		q := req.URL.Query()
		data, err := r.payload(atoiOr(q.Get("seq"), -1), atoiOr(q.Get("snap"), -1))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write(data)
	})

	return mux
}

// ListenAndServe serves the live view on addr (e.g. "localhost:8080"). It blocks, so run it in a goroutine.
func (r *Recorder) ListenAndServe(addr string) error {
	return http.ListenAndServe(addr, r.Handler())
}

func atoiOr(s string, def int) int {
	if v, err := strconv.Atoi(s); err == nil {
		return v
	}
	return def
}

// SaveHTML writes a page showing n's current weights and biases (no recorded activity) to path.
func SaveHTML(path string, n *network.Network) error {
	r, err := Wrap(n, Options{})
	if err != nil {
		return err
	}
	return r.SaveHTML(path)
}

// WriteHTML writes a page showing n's current weights and biases (no recorded activity) to w.
func WriteHTML(w io.Writer, n *network.Network) error {
	r, err := Wrap(n, Options{})
	if err != nil {
		return err
	}
	return r.WriteHTML(w)
}

// ---------------------------------------------------------------------------------------------------------------------
// Text output
// ---------------------------------------------------------------------------------------------------------------------

// Fprint writes a layer-by-layer summary of n to w: sizes, activation functions, and weight/bias statistics.
func Fprint(w io.Writer, n *network.Network) error {
	snap, err := Capture(n)
	if err != nil {
		return err
	}

	bw := bufio.NewWriter(w)

	sizes := []string{fmt.Sprint(snap.InputSize)}
	params := 0
	for _, l := range snap.Layers {
		sizes = append(sizes, fmt.Sprint(l.Size()))
		params += l.Size()*l.InputLen() + l.Size()
	}

	fmt.Fprintf(bw, "Network %s  (%d trainable parameters)\n\n", strings.Join(sizes, " → "), params)
	fmt.Fprintf(bw, "  [0] %-9s %5d neurons\n", snap.LayerName(0), snap.InputSize)

	for i, l := range snap.Layers {
		var ws []float64
		for _, row := range l.Weights {
			for _, x := range row {
				ws = append(ws, float64(x))
			}
		}
		bs := make([]float64, len(l.Biases))
		for j, b := range l.Biases {
			bs[j] = float64(b)
		}

		fmt.Fprintf(bw, "  [%d] %-9s %5d neurons  %-9s  weights %s\n", i+1, snap.LayerName(i+1), l.Size(), l.Activation, stats(ws))
		fmt.Fprintf(bw, "      %-9s %5s          %-9s  biases  %s\n", "", "", "", stats(bs))
	}

	return bw.Flush()
}

// FprintNeuron writes the bias and every incoming weight of a single neuron to w.
func FprintNeuron(w io.Writer, n *network.Network, layerIdx, neuronIdx int) error {
	snap, err := Capture(n)
	if err != nil {
		return err
	}

	bias, weights, err := snap.Neuron(layerIdx, neuronIdx)
	if err != nil {
		return err
	}

	bw := bufio.NewWriter(w)
	l := snap.Layers[layerIdx-1]

	fmt.Fprintf(bw, "%s layer (%d), neuron %d  ·  activation %s\n", snap.LayerName(layerIdx), layerIdx, neuronIdx, l.Activation)
	fmt.Fprintf(bw, "  bias     %+.6f\n", bias)
	fmt.Fprintf(bw, "  weights  %s\n\n", stats(weights))

	maxAbs := 0.0
	for _, x := range weights {
		if a := math.Abs(x); a > maxAbs && !math.IsInf(a, 0) {
			maxAbs = a
		}
	}

	const barWidth = 20
	from := snap.LayerName(layerIdx - 1)

	for j, x := range weights {
		bar := ""
		if maxAbs > 0 && !math.IsNaN(x) {
			k := int(math.Round(math.Min(math.Abs(x)/maxAbs, 1) * barWidth))
			if x < 0 {
				bar = strings.Repeat(" ", barWidth-k) + strings.Repeat("█", k) + "│"
			} else {
				bar = strings.Repeat(" ", barWidth) + "│" + strings.Repeat("█", k)
			}
		}
		fmt.Fprintf(bw, "  %s[%4d] → %+.6f  %s\n", from, j, x, bar)
	}

	return bw.Flush()
}

func stats(xs []float64) string {
	if len(xs) == 0 {
		return "(none)"
	}

	lo, hi, sum, bad := math.Inf(1), math.Inf(-1), 0.0, 0
	for _, x := range xs {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			bad++
			continue
		}
		lo, hi, sum = math.Min(lo, x), math.Max(hi, x), sum+x
	}

	if bad == len(xs) {
		return fmt.Sprintf("all %d values are NaN/Inf", bad)
	}

	s := fmt.Sprintf("min %+.4f  max %+.4f  mean %+.4f", lo, hi, sum/float64(len(xs)-bad))
	if bad > 0 {
		s += fmt.Sprintf("  (%d NaN/Inf)", bad)
	}
	return s
}

// ---------------------------------------------------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------------------------------------------------

const htmlTemplate = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neural Net Visualizer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0c0e11; --bg-2: #12151a; --panel: #151920; --panel-2: #1b2029; --ink: #e9e6df; --muted: #8a8f98; --faint: #5b616b;
    --line: #262c36; --line-2: #313844; --grid: rgba(255, 255, 255, 0.035);
    --pos: 62, 214, 197; --neg: 255, 122, 89; --node: #6f7782; --sel: #f5c542; --err: #ff5d5d;
    --glow: 0 0 0 1px rgba(245, 197, 66, .35), 0 0 24px rgba(245, 197, 66, .15);
    --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
    --sans: "IBM Plex Sans", ui-sans-serif, sans-serif;
    --serif: "Instrument Serif", Georgia, serif;
    color-scheme: dark;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f3f1ec; --bg-2: #ebe8e1; --panel: #fbfaf7; --panel-2: #f1eee8; --ink: #1c1d20; --muted: #6d6f74; --faint: #a3a39f;
      --line: #dedad2; --line-2: #cfcac0; --grid: rgba(0, 0, 0, 0.045);
      --pos: 13, 148, 136; --neg: 222, 83, 46; --node: #8a8c90; --sel: #c48a06; --err: #c62828;
      --glow: 0 0 0 1px rgba(196, 138, 6, .35), 0 0 18px rgba(196, 138, 6, .12);
      color-scheme: light;
    }
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  html, body { margin: 0; height: 100%; }
  body { background: var(--bg); color: var(--ink); font: 14px/1.5 var(--sans); display: flex; flex-direction: column;
         -webkit-font-smoothing: antialiased; }
  ::selection { background: rgba(var(--pos), .3); }

  /* ---- header ---- */
  header { display: flex; align-items: center; gap: 18px; padding: 14px 20px 12px; flex-wrap: wrap; animation: rise .5s ease both; }
  .brand { font: italic 30px/1 var(--serif); letter-spacing: -0.01em; margin: 0; font-weight: 400; }
  .brand b { font-style: normal; color: rgb(var(--pos)); font-weight: 400; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip { font: 500 12px/1 var(--mono); color: var(--muted); border: 1px solid var(--line); background: var(--bg-2);
          padding: 6px 9px; border-radius: 999px; letter-spacing: .02em; }
  .chip strong { color: var(--ink); font-weight: 500; }
  .legend { margin-left: auto; display: flex; align-items: center; gap: 10px; font: 12px var(--mono); color: var(--muted); }
  .ramp { width: 120px; height: 8px; border-radius: 4px; border: 1px solid var(--line);
          background: linear-gradient(90deg, rgb(var(--neg)), rgba(var(--neg), .08) 46%, rgba(var(--pos), .08) 54%, rgb(var(--pos))); }
  .legend .end { font-size: 11px; color: var(--faint); }
  #modeHelp { color: var(--ink); opacity: .75; }

  /* ---- transport ---- */
  .transport { display: flex; align-items: center; gap: 12px; margin: 0 20px 14px; padding: 8px 10px; flex-wrap: wrap;
               background: var(--panel); border: 1px solid var(--line); border-radius: 14px; animation: rise .5s .06s ease both; }
  .btn { font: 500 13px/1 var(--sans); color: var(--ink); background: var(--panel-2); border: 1px solid var(--line-2);
         border-radius: 9px; height: 32px; min-width: 32px; padding: 0 11px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
         transition: background .15s, border-color .15s, transform .08s; }
  .btn:hover { border-color: var(--muted); } .btn:active { transform: translateY(1px); }
  .btn.primary { background: rgb(var(--pos)); border-color: transparent; color: #071513; font-weight: 600; min-width: 84px; justify-content: center; }
  .btn.primary[aria-pressed="true"] { background: var(--sel); color: #1a1302; }
  .btn:focus-visible, .seg label:focus-within, input:focus-visible, select:focus-visible { outline: 2px solid var(--sel); outline-offset: 2px; }
  .scrubwrap { flex: 1; min-width: 160px; display: flex; flex-direction: column; gap: 3px; }
  #frameLabel { font: 12px var(--mono); color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #frameLabel b { color: var(--ink); font-weight: 500; }
  input[type=range] { -webkit-appearance: none; appearance: none; width: 100%; height: 18px; background: transparent; margin: 0; }
  input[type=range]::-webkit-slider-runnable-track { height: 4px; border-radius: 2px; background: var(--line-2); }
  input[type=range]::-moz-range-track { height: 4px; border-radius: 2px; background: var(--line-2); }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; margin-top: -5px; border-radius: 50%;
         background: var(--ink); border: 3px solid var(--panel); box-shadow: 0 0 0 1px var(--line-2); }
  input[type=range]::-moz-range-thumb { width: 10px; height: 10px; border-radius: 50%; background: var(--ink); border: 3px solid var(--panel); }
  .seg { display: inline-flex; background: var(--bg-2); border: 1px solid var(--line); border-radius: 10px; padding: 3px; gap: 2px; }
  .seg input { position: absolute; opacity: 0; pointer-events: none; }
  .seg label { font: 500 12px/1 var(--sans); color: var(--muted); padding: 7px 10px; border-radius: 7px; cursor: pointer; white-space: nowrap;
               transition: color .15s, background .15s; }
  .seg label:hover { color: var(--ink); }
  .seg input:checked + span { color: var(--ink); }
  .seg label:has(input:checked) { background: var(--panel-2); color: var(--ink); box-shadow: inset 0 0 0 1px var(--line-2); }
  .toggle { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); cursor: pointer; user-select: none; }
  .toggle input { accent-color: rgb(var(--pos)); }
  #live { font: 500 11px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--faint);
          display: inline-flex; align-items: center; gap: 6px; }
  #live:empty { display: none; }
  #live::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
  #live.on { color: rgb(var(--pos)); } #live.on::before { animation: pulse 1.6s ease-in-out infinite; }
  #live.off { color: var(--err); }

  /* ---- main ---- */
  main { flex: 1; display: flex; gap: 14px; min-height: 0; padding: 0 20px 20px; }
  #stage { flex: 1; position: relative; min-width: 0; border: 1px solid var(--line); border-radius: 16px; overflow: hidden;
           background:
             radial-gradient(120% 90% at 50% 40%, transparent 55%, rgba(0,0,0,.18) 100%),
             linear-gradient(var(--grid) 1px, transparent 1px) 0 0 / 24px 24px,
             linear-gradient(90deg, var(--grid) 1px, transparent 1px) 0 0 / 24px 24px,
             var(--bg-2);
           animation: rise .5s .12s ease both; }
  #net { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; }
  #tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--line-2); color: var(--ink);
         padding: 6px 9px; border-radius: 8px; font: 12px var(--mono); display: none; white-space: nowrap;
         box-shadow: 0 8px 24px rgba(0,0,0,.25); }
  aside { width: 400px; background: var(--panel); border: 1px solid var(--line); border-radius: 16px; display: flex; flex-direction: column;
          min-height: 0; overflow: hidden; animation: rise .5s .18s ease both; }
  .pick { padding: 14px 16px 12px; border-bottom: 1px solid var(--line); display: flex; gap: 10px; align-items: end; flex-wrap: wrap; }
  .field { display: flex; flex-direction: column; gap: 4px; font: 500 10.5px var(--mono); text-transform: uppercase; letter-spacing: .1em; color: var(--faint); }
  .field select, .field input { font: 13px var(--mono); color: var(--ink); background: var(--bg-2); border: 1px solid var(--line-2);
         border-radius: 8px; height: 32px; padding: 0 8px; text-transform: none; letter-spacing: 0; }
  .field input { width: 84px; }
  .hint { color: var(--faint); font-size: 11.5px; width: 100%; }
  kbd { font: 10.5px var(--mono); border: 1px solid var(--line-2); border-bottom-width: 2px; border-radius: 4px; padding: 0 4px; color: var(--muted); }
  #panel { overflow: auto; flex: 1; padding: 0 16px 18px; scrollbar-width: thin; scrollbar-color: var(--line-2) transparent; }
  section { padding-top: 16px; }
  section + section { border-top: 1px dashed var(--line-2); margin-top: 16px; }
  .eyebrow { font: 500 10.5px var(--mono); text-transform: uppercase; letter-spacing: .12em; color: rgb(var(--pos)); margin-bottom: 2px; }
  h2 { font: 400 24px/1.1 var(--serif); margin: 0 0 4px; letter-spacing: -0.005em; }
  h3 { font: 500 10.5px var(--mono); text-transform: uppercase; letter-spacing: .12em; color: var(--muted); margin: 18px 0 8px; }
  .muted { color: var(--muted); }
  .kv { display: grid; grid-template-columns: repeat(auto-fill, minmax(108px, 1fr)); gap: 6px; margin: 10px 0; }
  .kv > div { background: var(--bg-2); border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; min-width: 0; }
  .kv dt { font: 500 10px var(--mono); text-transform: uppercase; letter-spacing: .1em; color: var(--faint); margin-bottom: 3px;
           white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .kv dd { margin: 0; font: 500 13px var(--mono); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .kv .big { font-size: 17px; color: var(--sel); }
  .kv > div.wide { grid-column: 1 / -1; }
  .err { color: var(--err); font: 12px var(--mono); background: rgba(255, 93, 93, .08); border: 1px solid rgba(255, 93, 93, .25);
         padding: 8px 10px; border-radius: 8px; }
  .note { color: var(--muted); font-size: 13px; }
  canvas.grid { display: block; image-rendering: pixelated; border-radius: 10px; margin: 4px 0 8px; background: var(--bg);
                box-shadow: 0 0 0 1px var(--line); }
  .sorter { display: flex; gap: 8px; align-items: center; margin: 12px 0 6px; }
  .sorter .btn { height: 26px; font-size: 12px; padding: 0 9px; }
  .sorter .btn[aria-pressed="true"] { border-color: rgb(var(--pos)); color: rgb(var(--pos)); }
  table { width: 100%; border-collapse: collapse; font: 12px/1.3 var(--mono); }
  th { text-align: right; font-weight: 500; font-size: 10px; text-transform: uppercase; letter-spacing: .1em; color: var(--faint);
       padding: 6px 6px; border-bottom: 1px solid var(--line-2); position: sticky; top: 0; background: var(--panel); }
  th:first-child { text-align: left; }
  td { padding: 4px 6px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
  td:first-child { text-align: left; color: var(--muted); }
  tbody tr { transition: background .1s; }
  tbody tr:hover td { background: var(--panel-2); }
  tr.hl td { color: var(--sel); } tr.hl td:first-child { color: var(--sel); }
  .bar { position: relative; height: 8px; min-width: 60px; border-radius: 4px; background: var(--bg-2); }
  .bar::before { content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; border-left: 1px solid var(--line-2); }
  .bar span { position: absolute; top: 0; bottom: 0; border-radius: 4px; }

  @keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
  @keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(var(--pos), .6); } 50% { box-shadow: 0 0 0 5px rgba(var(--pos), 0); } }
  @media (prefers-reduced-motion: reduce) { *, *::before { animation: none !important; transition: none !important; } }
  @media (max-width: 860px) {
    header { padding: 12px 16px 10px; gap: 10px 14px; }
    .legend { margin-left: 0; flex-wrap: wrap; min-width: 0; }
    #modeHelp { flex-basis: 100%; white-space: normal; }
    .transport { margin: 0 16px 12px; gap: 10px; }
    .scrubwrap { order: 5; flex-basis: 100%; min-width: 0; }
    .seg { max-width: 100%; overflow-x: auto; scrollbar-width: none; }
    main { flex-direction: column; padding: 0 16px 16px; }
    #stage { min-height: 62vh; flex: none; }
    aside { width: 100%; flex: none; max-height: none; }
    #panel { overflow: visible; }
  }
</style>
</head>
<body>
<header>
  <h1 class="brand">neural<b>·</b>net</h1>
  <div class="chips" id="summary"></div>
  <div class="legend">
    <span class="end">−</span><span class="ramp" aria-hidden="true"></span><span class="end">+</span>
    <span id="modeHelp"></span>
  </div>
</header>
<div class="transport">
  <button class="btn" id="prev" title="Previous frame (,)" aria-label="Previous frame">◀</button>
  <button class="btn primary" id="play" title="Play / pause (space)">▶ Play</button>
  <button class="btn" id="next" title="Next frame (.)" aria-label="Next frame">▶</button>
  <div class="scrubwrap">
    <span id="frameLabel"></span>
    <input id="scrub" type="range" min="0" max="0" value="0" aria-label="Frame">
  </div>
  <div class="seg" id="mode" role="radiogroup" aria-label="View">
    <label><input type="radio" name="mode" value="auto" checked>Auto</label>
    <label><input type="radio" name="mode" value="weights">Weights</label>
    <label><input type="radio" name="mode" value="signal">Signal w·a</label>
    <label><input type="radio" name="mode" value="update">Update ΔW</label>
  </div>
  <label class="toggle"><input id="anim" type="checkbox" checked> Animate</label>
  <label class="toggle" id="followWrap" hidden><input id="follow" type="checkbox" checked> Follow latest</label>
  <span id="live"></span>
</div>
<main>
  <div id="stage"><canvas id="net"></canvas><div id="tip"></div></div>
  <aside>
    <div class="pick">
      <label class="field">Layer <select id="pickLayer"></select></label>
      <label class="field">Neuron <input id="pickNeuron" type="number" min="0" value="0"></label>
      <div class="hint">Click a neuron · <kbd>←</kbd><kbd>→</kbd> layer · <kbd>↑</kbd><kbd>↓</kbd> neuron · <kbd>,</kbd><kbd>.</kbd> frame · <kbd>space</kbd> play · <kbd>esc</kbd> clear</div>
    </div>
    <div id="panel"><section id="frameBox"></section><section id="detail"></section></div>
  </aside>
</main>
<script id="net-data" type="application/json">/*__DATA__*/null</script>
<script>
(function () {
  "use strict";

  const EMBED = JSON.parse(document.getElementById("net-data").textContent);
  const LIVE = EMBED === null;
  const toNum = (x) => typeof x === "number" ? x : Number(x);
  const nums = (a) => a ? a.map(toNum) : null;
  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  const finiteMax = (arr) => { let m = 0; for (const v of arr) if (isFinite(v)) m = Math.max(m, Math.abs(v)); return m; };
  const fmt = (x) => x == null ? "—" : isFinite(x) ? (x >= 0 ? "+" : "") + x.toFixed(5) : String(x);
  const $ = (id) => document.getElementById(id);

  // ---- data ---------------------------------------------------------------------------------------------------------
  let layers = [];            // [{name, size, activation, x, top, gap, r}]
  let maxFrames = 0;
  const snaps = new Map();    // id -> [{W, b}] for each non-input layer
  let frames = [];
  let lastSeq = -1, lastSnap = -1, cur = 0;

  function ingest(d) {
    if (d.meta && layers.length === 0) {
      layers = [{ name: "Input", size: d.meta.inputSize, activation: null }];
      d.meta.layers.forEach((l, i) => layers.push({
        name: i === d.meta.layers.length - 1 ? "Output" : "Hidden " + (i + 1), size: l.size, activation: l.activation,
      }));
      const params = d.meta.layers.reduce((s, l, i) => s + l.size * layers[i].size + l.size, 0);
      const chips = $("summary");
      chips.innerHTML = "";
      const chip = (label, value) => { const c = el("span", label ? label + " " : "", "chip"); c.append(el("strong", value)); chips.append(c); };
      chip("", layers.map((l) => l.size).join(" → "));
      chip("params", params.toLocaleString());
      buildPickers();
      layout();
    }
    maxFrames = d.maxFrames || maxFrames;
    for (const id in d.snapshots || {}) {
      snaps.set(+id, d.snapshots[id].layers.map((l) => ({ W: l.weights.map(nums), b: nums(l.biases) })));
      lastSnap = Math.max(lastSnap, +id);
    }
    for (const f of d.frames || []) {
      frames.push({
        seq: f.seq, kind: f.kind, label: f.label, snap: f.snap, after: f.after == null ? null : f.after, err: f.err || "",
        input: nums(f.input), z: f.z ? f.z.map(nums) : null, a: f.a ? f.a.map(nums) : null,
      });
      lastSeq = Math.max(lastSeq, f.seq);
    }
    if (maxFrames && frames.length > maxFrames) frames = frames.slice(frames.length - maxFrames);
    const used = new Set([lastSnap]);
    frames.forEach((f) => { used.add(f.snap); if (f.after != null) used.add(f.after); });
    for (const id of [...snaps.keys()]) if (!used.has(id)) snaps.delete(id);
  }

  const act = (f, li) => li === 0 ? f.input : (f.a && f.a[li - 1]) || null;
  const hasTrace = (f) => !!(f.a && f.a.length);
  const hasUpdate = (f) => f.after != null;
  const changed = (f) => f.after != null && f.after !== f.snap;

  function effMode(f) {
    const m = modeValue();
    if (m === "signal" && hasTrace(f)) return "signal";
    if (m === "update" && hasUpdate(f)) return "update";
    if (m === "auto") return changed(f) ? "update" : hasTrace(f) ? "signal" : "weights";
    return "weights";
  }
  const MODE_HELP = {
    weights: "edges = weight w · nodes = bias b",
    signal: "edges = w·a (signal carried) · nodes = activation a",
    update: "edges = ΔW · nodes = Δb (change applied by Train)",
  };

  // Edge values between layer li-1 and li, for the given view. get(i, j): j in li-1 → i in li.
  function gapVals(f, mode, li) {
    f._g = f._g || {};
    const key = mode + li;
    if (f._g[key]) return f._g[key];
    const S = snaps.get(f.snap)[li - 1];
    let get = (i, j) => S.W[i][j];
    if (mode === "signal" && act(f, li - 1)) { const A = act(f, li - 1); get = (i, j) => S.W[i][j] * A[j]; }
    if (mode === "update") { const T = snaps.get(f.after)[li - 1]; get = (i, j) => T.W[i][j] - S.W[i][j]; }
    let max = 0;
    for (let i = 0; i < layers[li].size; i++) for (let j = 0; j < layers[li - 1].size; j++) {
      const v = get(i, j); if (isFinite(v)) max = Math.max(max, Math.abs(v));
    }
    return (f._g[key] = { get, max });
  }

  // Node values for a layer in the given view.
  function nodeVals(f, mode, li) {
    f._n = f._n || {};
    const key = mode + li;
    if (f._n[key]) return f._n[key];
    let vals = null;
    if (mode === "signal") vals = act(f, li);
    else if (li > 0) {
      const S = snaps.get(f.snap)[li - 1];
      if (mode === "update") { const T = snaps.get(f.after)[li - 1]; vals = S.b.map((b, i) => T.b[i] - b); }
      else vals = S.b;
    }
    return (f._n[key] = { vals, max: vals ? finiteMax(vals) : 0 });
  }

  // ---- layout + drawing ---------------------------------------------------------------------------------------------
  const canvas = $("net"), ctx = canvas.getContext("2d"), tip = $("tip");
  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  const POS = color("--pos"), NEG = color("--neg");
  const MONO = "'IBM Plex Mono', ui-monospace, monospace";
  const modeValue = () => (document.querySelector("input[name=mode]:checked") || { value: "auto" }).value;
  const rgba = (v, a) => "rgba(" + (v < 0 ? NEG : POS) + "," + a + ")";
  let W = 0, H = 0, dpr = 1;
  const PAD_TOP = 56, PAD_BOTTOM = 16;

  function layout() {
    const r = canvas.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    W = r.width; H = r.height;
    canvas.width = Math.max(1, Math.round(W * dpr)); canvas.height = Math.max(1, Math.round(H * dpr));
    const n = layers.length, usable = H - PAD_TOP - PAD_BOTTOM;
    const PAD_X = Math.max(40, Math.min(84, W * 0.09));
    layers.forEach((l, i) => {
      l.x = n === 1 ? W / 2 : PAD_X + i * (W - 2 * PAD_X) / (n - 1);
      l.gap = Math.min(40, usable / l.size);
      l.top = PAD_TOP + (usable - l.gap * l.size) / 2 + l.gap / 2;
      l.r = Math.max(0.8, Math.min(11, l.gap * 0.38));
    });
    baseKey = ""; ghostKey = "";
    draw();
  }
  const ny = (l, i) => l.top + i * l.gap;

  // One offscreen canvas per gap so the forward/backward sweep can fade gaps in one at a time.
  // Faint outline of every connection, drawn under the signal/update views so the network is always visible.
  let ghost = null, ghostKey = "";
  function ensureGhost(f) {
    const key = f.snap + "|" + canvas.width + "x" + canvas.height;
    if (key === ghostKey) return;
    ghostKey = key;
    ghost = document.createElement("canvas");
    ghost.width = canvas.width; ghost.height = canvas.height;
    const b = ghost.getContext("2d"); b.scale(dpr, dpr);
    b.strokeStyle = color("--node"); b.lineWidth = 0.5;
    for (let li = 1; li < layers.length; li++) {
      const l = layers[li], p = layers[li - 1], path = new Path2D();
      for (let i = 0; i < l.size; i++) for (let j = 0; j < p.size; j++) { path.moveTo(p.x, ny(p, j)); path.lineTo(l.x, ny(l, i)); }
      b.globalAlpha = Math.min(0.35, 0.35 * Math.sqrt(256 / (l.size * p.size)) + 0.04);
      b.stroke(path);
    }
  }

  let bases = [], baseKey = "";
  function ensureBases(f, mode) {
    const key = f.seq + "|" + mode + "|" + canvas.width + "x" + canvas.height;
    if (key === baseKey) return;
    baseKey = key; bases = [null];
    const BUCKETS = 10;
    for (let li = 1; li < layers.length; li++) {
      const c = document.createElement("canvas");
      c.width = canvas.width; c.height = canvas.height;
      const b = c.getContext("2d"); b.scale(dpr, dpr);
      const l = layers[li], p = layers[li - 1], g = gapVals(f, mode, li);
      if (g.max > 0) {
        const scale = Math.min(0.9, 0.9 * Math.sqrt(400 / (l.size * p.size)) + 0.12);
        const pos = [], neg = [];
        for (let k = 0; k < BUCKETS; k++) { pos.push(new Path2D()); neg.push(new Path2D()); }
        for (let i = 0; i < l.size; i++) {
          const yi = ny(l, i);
          for (let j = 0; j < p.size; j++) {
            const v = g.get(i, j);
            if (!isFinite(v) || v === 0) continue;
            const k = Math.min(BUCKETS - 1, Math.floor(Math.abs(v) / g.max * BUCKETS));
            const path = (v < 0 ? neg : pos)[k];
            path.moveTo(p.x, ny(p, j)); path.lineTo(l.x, yi);
          }
        }
        b.lineWidth = 0.7;
        for (let k = 0; k < BUCKETS; k++) {
          const a = ((k + 1) / BUCKETS) * scale;
          b.strokeStyle = rgba(1, a); b.stroke(pos[k]);
          b.strokeStyle = rgba(-1, a); b.stroke(neg[k]);
        }
      }
      bases.push(c);
    }
  }

  // Sweep animation: forward passes reveal input → output, updates reveal output → input (the backprop direction).
  let reveal = Infinity, animDir = 1, animToken = 0;
  const lastLayer = () => layers.length - 1;
  function gapAlpha(li) {
    if (reveal === Infinity) return 1;
    return animDir > 0 ? clamp01(reveal - (li - 1)) : clamp01(reveal - (lastLayer() - li));
  }
  const nodeLit = (li) => li === 0 && animDir > 0 ? 1 : gapAlpha(Math.max(li, 1));
  function animate() {
    const f = frames[cur];
    if (!f) return;
    const mode = effMode(f);
    animDir = mode === "update" ? -1 : 1;
    const token = ++animToken;
    if (!$("anim").checked || mode === "weights") { reveal = Infinity; draw(); return; }
    const L = lastLayer(), dur = 280 * L, start = performance.now();
    reveal = 0;
    const step = (t) => {
      if (token !== animToken) return;
      reveal = (t - start) / dur * L;
      if (reveal >= L) reveal = Infinity;
      draw();
      if (reveal !== Infinity) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  let sel = null, hover = null, hoverEdge = null;

  function draw() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const f = frames[cur];
    if (!layers.length || !f || !snaps.has(f.snap)) return;
    const mode = effMode(f);
    ensureBases(f, mode);
    if (mode !== "weights") {
      ensureGhost(f);
      ctx.globalAlpha = sel ? 0.5 : 1;
      ctx.drawImage(ghost, 0, 0);
    }

    for (let li = 1; li < layers.length; li++) {
      ctx.globalAlpha = gapAlpha(li) * (sel ? 0.15 : 1);
      if (ctx.globalAlpha > 0) ctx.drawImage(bases[li], 0, 0);
    }
    ctx.globalAlpha = 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (sel) drawSelected(f, mode, sel);

    const PANEL = color("--panel"), NODE = color("--node"), MUTED = color("--muted"), INK = color("--ink");
    layers.forEach((l, li) => {
      const nv = nodeVals(f, mode, li), lit = nodeLit(li);
      const out = li === lastLayer() && mode === "signal" && nv.vals;
      let argmax = -1;
      if (out) nv.vals.forEach((v, i) => { if (argmax < 0 || v > nv.vals[argmax]) argmax = i; });
      for (let i = 0; i < l.size; i++) {
        const y = ny(l, i), v = nv.vals ? nv.vals[i] : null;
        const t = v != null && isFinite(v) && nv.max > 0 ? clamp01(Math.abs(v) / nv.max) * lit : 0;
        if (l.r < 2) {
          const h = Math.max(1, l.r * 2);
          ctx.fillStyle = PANEL; ctx.fillRect(l.x - 4, y - h / 2, 8, h);
          ctx.fillStyle = t > 0 ? rgba(v, 0.15 + 0.85 * t) : NODE;
          ctx.globalAlpha = t > 0 ? 1 : 0.35;
          ctx.fillRect(l.x - 4, y - h / 2, 8, h);
          ctx.globalAlpha = 1;
        } else {
          ctx.beginPath(); ctx.arc(l.x, y, l.r, 0, Math.PI * 2);
          ctx.fillStyle = PANEL; ctx.fill();
          if (t > 0) {
            ctx.save(); ctx.shadowColor = rgba(v, 0.8 * t); ctx.shadowBlur = 14 * t;
            ctx.fillStyle = rgba(v, t); ctx.fill(); ctx.restore();
          }
          if (v != null && !isFinite(v)) { ctx.fillStyle = color("--err"); ctx.fill(); }
          ctx.strokeStyle = t > 0 ? rgba(v, 0.35 + 0.65 * t) : NODE; ctx.lineWidth = 1.25; ctx.stroke();
        }
        if (out && l.r >= 3 && lit > 0) {
          ctx.fillStyle = i === argmax ? color("--sel") : MUTED;
          ctx.font = (i === argmax ? "600 " : "") + "11px " + MONO;
          ctx.textAlign = "left";
          ctx.fillText(i + ": " + (isFinite(v) ? v.toFixed(3) : String(v)), l.x + l.r + 6, y + 4);
        }
      }
      ctx.textAlign = "center";
      ctx.fillStyle = INK; ctx.font = "500 11px " + MONO;
      if ("letterSpacing" in ctx) ctx.letterSpacing = "1.5px";
      const inside = (txt) => { const hw = ctx.measureText(txt).width / 2 + 8; return Math.max(hw, Math.min(W - hw, l.x)); };
      ctx.fillText(l.name.toUpperCase(), inside(l.name.toUpperCase()), 22);
      ctx.fillStyle = MUTED; ctx.font = "11px " + MONO;
      const sub = l.size + (l.activation ? " · " + l.activation : "");
      ctx.fillText(sub, inside(sub), 38);
      if ("letterSpacing" in ctx) ctx.letterSpacing = "0px";
    });

    const ring = (p, c, w) => {
      if (!p) return;
      const l = layers[p.l], r = Math.max(l.r, 3) + 3;
      ctx.strokeStyle = c; ctx.lineWidth = w;
      ctx.beginPath(); ctx.arc(l.x, ny(l, p.i), r, 0, Math.PI * 2); ctx.stroke();
    };
    const SEL = color("--sel");
    if (hoverEdge) {
      const l = layers[hoverEdge.l], p = layers[hoverEdge.l - 1], v = gapVals(f, mode, hoverEdge.l).get(hoverEdge.i, hoverEdge.j);
      ctx.strokeStyle = rgba(v, 1); ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(p.x, ny(p, hoverEdge.j)); ctx.lineTo(l.x, ny(l, hoverEdge.i)); ctx.stroke();
      ring({ l: hoverEdge.l - 1, i: hoverEdge.j }, SEL, 1.5);
      ring({ l: hoverEdge.l, i: hoverEdge.i }, SEL, 1.5);
    }
    ring(hover, MUTED, 1.5);
    ring(sel, SEL, 2.5);

    if (mode === "update" && !changed(f)) {
      const msg = "No weights or biases changed during this call";
      ctx.font = "12px " + MONO; ctx.textAlign = "center";
      const w = ctx.measureText(msg).width + 28, x = W / 2, y = H - 30;
      ctx.fillStyle = PANEL; ctx.strokeStyle = color("--line-2"); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x - w / 2, y - 16, w, 28, 14) : ctx.rect(x - w / 2, y - 16, w, 28);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = MUTED; ctx.fillText(msg, x, y + 2);
    }
  }

  // Incoming (strong) and outgoing (faint) edges of the selected neuron, normalised to that neuron.
  function drawSelected(f, mode, s) {
    const l = layers[s.l], y = ny(l, s.i);
    const lines = (li, n, get, alpha) => {
      const g = gapVals(f, mode, li);
      let m = 0;
      for (let k = 0; k < n; k++) { const v = get(g, k); if (isFinite(v)) m = Math.max(m, Math.abs(v)); }
      if (!m) return;
      for (let k = 0; k < n; k++) {
        const v = get(g, k); if (!isFinite(v) || v === 0) continue;
        const t = Math.abs(v) / m;
        ctx.strokeStyle = rgba(v, alpha * (0.15 + 0.85 * t)); ctx.lineWidth = 0.5 + 2 * t;
        const pts = li === s.l ? [layers[li - 1].x, ny(layers[li - 1], k), l.x, y] : [l.x, y, layers[li].x, ny(layers[li], k)];
        ctx.beginPath(); ctx.moveTo(pts[0], pts[1]); ctx.lineTo(pts[2], pts[3]); ctx.stroke();
      }
    };
    if (s.l > 0) lines(s.l, layers[s.l - 1].size, (g, j) => g.get(s.i, j), gapAlpha(s.l));
    if (s.l < lastLayer()) lines(s.l + 1, layers[s.l + 1].size, (g, k) => g.get(k, s.i), 0.5 * gapAlpha(s.l + 1));
  }

  // ---- interaction --------------------------------------------------------------------------------------------------
  function hit(px, py) {
    let best = -1, bestDx = Infinity;
    layers.forEach((l, li) => { const dx = Math.abs(px - l.x); if (dx < bestDx) { bestDx = dx; best = li; } });
    if (best < 0 || bestDx > 18) return null;
    const l = layers[best], i = Math.round((py - l.top) / l.gap);
    if (i < 0 || i >= l.size || Math.abs(py - ny(l, i)) > Math.max(l.gap / 2, l.r + 4)) return null;
    return { l: best, i };
  }
  const same = (a, b) => (!a && !b) || (a && b && a.l === b.l && a.i === b.i);

  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect(), h = hit(e.clientX - r.left, e.clientY - r.top);
    if (!same(h, hover)) { hover = h; draw(); }
    const f = frames[cur];
    if (!h || !f) { tip.style.display = "none"; return; }
    const mode = effMode(f), v = nodeVals(f, mode, h.l).vals;
    const what = { weights: "b", signal: "a", update: "Δb" }[mode];
    tip.textContent = layers[h.l].name + " #" + h.i + (v ? "   " + what + " = " + fmt(v[h.i]) : "");
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX - r.left + 12, W - tip.offsetWidth - 4) + "px";
    tip.style.top = (e.clientY - r.top + 12) + "px";
  });
  canvas.addEventListener("mouseleave", () => { hover = null; tip.style.display = "none"; draw(); });
  canvas.addEventListener("click", (e) => {
    const r = canvas.getBoundingClientRect();
    select(hit(e.clientX - r.left, e.clientY - r.top));
  });

  const pickLayer = $("pickLayer"), pickNeuron = $("pickNeuron");
  function buildPickers() {
    pickLayer.innerHTML = "";
    layers.forEach((l, i) => {
      const o = document.createElement("option");
      o.value = i; o.textContent = i + " · " + l.name + " (" + l.size + ")";
      pickLayer.appendChild(o);
    });
  }
  pickLayer.addEventListener("change", () => {
    const l = +pickLayer.value;
    select({ l, i: Math.min(+pickNeuron.value || 0, layers[l].size - 1) });
  });
  pickNeuron.addEventListener("input", () => {
    const l = +pickLayer.value, i = +pickNeuron.value;
    if (Number.isInteger(i) && i >= 0 && i < layers[l].size) select({ l, i });
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type !== "checkbox" && e.target.type !== "range") return;
    if (e.target.tagName === "SELECT") return;
    if (e.key === "Escape") { select(null); return; }
    if (e.key === ",") { setFrame(cur - 1); return; }
    if (e.key === ".") { setFrame(cur + 1); return; }
    if (e.key === " ") { e.preventDefault(); togglePlay(); return; }
    if (!sel) return;
    let l = sel.l, i = sel.i;
    if (e.key === "ArrowUp") i--; else if (e.key === "ArrowDown") i++;
    else if (e.key === "ArrowLeft") l--; else if (e.key === "ArrowRight") l++;
    else return;
    e.preventDefault();
    l = Math.max(0, Math.min(lastLayer(), l));
    i = Math.max(0, Math.min(layers[l].size - 1, i));
    select({ l, i });
  });

  function select(s) {
    sel = s; hoverEdge = null;
    if (s) { pickLayer.value = s.l; pickNeuron.value = s.i; pickNeuron.max = layers[s.l].size - 1; }
    draw();
    renderDetail();
  }

  // ---- timeline -----------------------------------------------------------------------------------------------------
  const scrub = $("scrub");
  let playTimer = null;

  function setFrame(k, anim) {
    if (!frames.length) return;
    cur = Math.max(0, Math.min(frames.length - 1, k));
    updateTimeline();
    renderFrame(); renderDetail();
    if (anim === false) draw(); else animate();
  }
  function updateTimeline() {
    scrub.max = Math.max(0, frames.length - 1); scrub.value = cur;
    const f = frames[cur];
    const fl = $("frameLabel");
    fl.textContent = "";
    if (f) { fl.append(el("b", f.label), " · frame " + (cur + 1) + " of " + frames.length); } else fl.textContent = "no frames";
    $("modeHelp").textContent = f ? MODE_HELP[effMode(f)] : "";
  }
  scrub.addEventListener("input", () => setFrame(+scrub.value, false));
  $("prev").onclick = () => setFrame(cur - 1);
  $("next").onclick = () => setFrame(cur + 1);
  $("mode").addEventListener("change", () => { updateTimeline(); renderFrame(); renderDetail(); animate(); });
  $("play").onclick = togglePlay;
  function togglePlay() {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    else {
      if (cur >= frames.length - 1 && !LIVE) setFrame(0);
      playTimer = setInterval(() => {
        if (cur >= frames.length - 1) { if (!LIVE) togglePlay(); return; }
        setFrame(cur + 1);
      }, Math.max(400, 300 * lastLayer() + 300));
    }
    $("play").textContent = playTimer ? "❚❚ Pause" : "▶ Play";
    $("play").setAttribute("aria-pressed", String(!!playTimer));
  }

  // ---- side panel ---------------------------------------------------------------------------------------------------
  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text != null && text !== "") e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }
  function kv(pairs) {
    const d = el("dl", "", "kv");
    for (const p of pairs) {
      const cell = el("div", "", p[2] === "big" ? "wide" : "");
      cell.append(el("dt", p[0]), el("dd", p[1], p[2]));
      d.append(cell);
    }
    return d;
  }
  // Square grid image of values (e.g. 784 MNIST pixels → 28×28). onHover(k) / onLeave() optional.
  function grid(vals, signed, onHover, onLeave) {
    const side = Math.round(Math.sqrt(vals.length));
    if (side * side !== vals.length || side < 3) return null;
    const c = el("canvas", "", "grid");
    c.width = c.height = side;
    const px = Math.max(3, Math.floor(196 / side));
    c.style.width = c.style.height = side * px + "px";
    const g = c.getContext("2d"), img = g.createImageData(side, side), m = finiteMax(vals) || 1;
    const pos = POS.split(",").map(Number), neg = NEG.split(",").map(Number), ink = signed ? null : [128, 128, 128];
    vals.forEach((v, k) => {
      const t = isFinite(v) ? clamp01(Math.abs(v) / m) : 0, rgb = ink || (v < 0 ? neg : pos);
      img.data.set([rgb[0], rgb[1], rgb[2], Math.round(255 * (0.05 + 0.95 * t))], k * 4);
    });
    g.putImageData(img, 0, 0);
    if (onHover) {
      c.addEventListener("mousemove", (e) => {
        const r = c.getBoundingClientRect();
        const k = Math.floor((e.clientY - r.top) / px) * side + Math.floor((e.clientX - r.left) / px);
        if (k >= 0 && k < vals.length) { c.title = "#" + k + "  " + fmt(vals[k]); onHover(k); }
      });
      c.addEventListener("mouseleave", onLeave);
    }
    return c;
  }

  function barCell(v, m) {
    const td = el("td"), bar = el("div", "", "bar"), span = el("span");
    const pct = isFinite(v) && m ? 50 * clamp01(Math.abs(v) / m) : 0;
    span.style.background = rgba(v, 0.9); span.style.width = pct + "%"; span.style.left = (v < 0 ? 50 - pct : 50) + "%";
    bar.append(span); td.append(bar); td.style.width = "30%";
    return td;
  }

  function renderFrame() {
    const box = $("frameBox");
    box.innerHTML = "";
    const f = frames[cur];
    if (!f) { box.append(el("p", LIVE ? "Waiting for the network to be called…" : "Nothing recorded.", "note")); return; }
    const kinds = { snapshot: "Weights snapshot", forward: "Forward pass · Categorize()", train: "Training step · Train()" };
    box.append(el("div", kinds[f.kind] || f.kind, "eyebrow"), el("h2", f.label));
    if (f.err) box.append(el("p", "Error: " + f.err, "err"));
    const want = modeValue();
    if (want === "signal" && !hasTrace(f)) box.append(el("p", "This frame has no forward trace; showing weights.", "note"));
    if (want === "update" && !hasUpdate(f)) box.append(el("p", "This frame isn't a Train() call; showing weights.", "note"));

    if (hasUpdate(f)) {
      const S = snaps.get(f.snap), T = snaps.get(f.after);
      let changed = 0, total = 0, mw = 0, mb = 0;
      S.forEach((l, k) => {
        l.W.forEach((row, i) => row.forEach((w, j) => {
          const d = T[k].W[i][j] - w; total++;
          if (d !== 0) changed++;
          if (isFinite(d)) mw = Math.max(mw, Math.abs(d));
        }));
        l.b.forEach((b, i) => { const d = T[k].b[i] - b; if (isFinite(d)) mb = Math.max(mb, Math.abs(d)); });
      });
      box.append(el("h3", "Update applied by Train()"));
      box.append(kv([
        ["weights changed", changed.toLocaleString() + " / " + total.toLocaleString()],
        ["max |ΔW|", mw.toExponential(3)],
        ["max |Δb|", mb.toExponential(3)],
      ]));
    }

    if (f.input) {
      const g = grid(f.input, false);
      if (g) { box.append(el("h3", "Input")); box.append(g); }
    }

    const out = act(f, lastLayer());
    if (out) {
      box.append(el("h3", "Output"));
      let am = 0; out.forEach((v, i) => { if (v > out[am]) am = i; });
      const m = finiteMax(out) || 1;
      const t = el("table"), tb = el("tbody");
      out.forEach((v, i) => {
        const tr = el("tr", "", i === am ? "hl" : "");
        tr.append(el("td", "#" + i + (i === am ? "  ← max" : "")), el("td", fmt(v)), barCell(v, m));
        tr.onclick = () => select({ l: lastLayer(), i });
        tb.append(tr);
      });
      t.append(tb); box.append(t);
    }
  }

  let sortBy = "index";

  function renderDetail() {
    const box = $("detail");
    box.innerHTML = "";
    const f = frames[cur];
    if (!sel || !f) { box.append(el("p", "Click a neuron to see its bias and incoming weights.", "note")); return; }
    const mode = effMode(f), l = layers[sel.l];
    box.append(el("div", l.name + " layer" + (l.activation ? " · " + l.activation : ""), "eyebrow"), el("h2", "Neuron " + sel.i));

    if (sel.l === 0) {
      box.append(el("p", "Input neurons hold the raw input; they have no bias or incoming weights. Showing outgoing connections.", "muted"));
      if (f.input) box.append(kv([["input value", fmt(f.input[sel.i]), "big"]]));
    } else {
      const S = snaps.get(f.snap)[sel.l - 1], T = hasUpdate(f) ? snaps.get(f.after)[sel.l - 1] : null;
      const pairs = [["bias b", fmt(S.b[sel.i]), "big"]];
      if (T) pairs.push(["Δb (Train)", fmt(T.b[sel.i] - S.b[sel.i])]);
      pairs.push(["activation", l.activation]);
      if (f.z && f.z[sel.l - 1]) pairs.push(["z = W·a + b", fmt(f.z[sel.l - 1][sel.i])]);
      const a = act(f, sel.l);
      if (a) pairs.push(["a = " + l.activation + "(z)", fmt(a[sel.i])]);
      box.append(kv(pairs));
    }

    // Connection rows: incoming for hidden/output neurons, outgoing for input neurons.
    const incoming = sel.l > 0, li = incoming ? sel.l : 1;
    if (li > lastLayer()) return;
    const S = snaps.get(f.snap)[li - 1], T = hasUpdate(f) ? snaps.get(f.after)[li - 1] : null;
    const A = act(f, li - 1), g = gapVals(f, mode, li), other = layers[incoming ? li - 1 : li];
    const rows = [];
    for (let k = 0; k < other.size; k++) {
      const i = incoming ? sel.i : k, j = incoming ? k : sel.i, w = S.W[i][j];
      rows.push({ k, w, a: A ? A[j] : null, wa: A ? w * A[j] : null, dw: T ? T.W[i][j] - w : null, v: g.get(i, j), edge: { l: li, i, j } });
    }

    box.append(el("h3", (incoming ? "Incoming from " : "Outgoing to ") + other.name + " (" + rows.length + ")"));
    const vals = rows.map((r) => r.v), fin = vals.filter(isFinite);
    box.append(kv([
      ["showing", { weights: "w", signal: "w·a", update: "Δw" }[mode]],
      ["min", fin.length ? fmt(Math.min(...fin)) : "—"],
      ["max", fin.length ? fmt(Math.max(...fin)) : "—"],
    ]));
    const gr = grid(vals, true, (k) => { hoverEdge = rows[k].edge; draw(); }, () => { hoverEdge = null; draw(); });
    if (gr) box.append(gr);

    const sorter = el("div", "", "sorter");
    sorter.append(el("span", "Sort"));
    [["index", "by index"], ["abs", "by |shown value|"]].forEach((o) => {
      const b = el("button", o[1], "btn"); b.setAttribute("aria-pressed", String(sortBy === o[0]));
      b.onclick = () => { sortBy = o[0]; renderDetail(); };
      sorter.append(b);
    });
    box.append(sorter);

    const order = rows.slice();
    if (sortBy === "abs") order.sort((x, y) => (Math.abs(y.v) || 0) - (Math.abs(x.v) || 0));
    const m = finiteMax(vals) || 1;
    const t = el("table"), head = el("tr");
    const cols = [["w", "w"]];
    if (A) cols.push(["a", incoming ? "a in" : "a"], ["wa", "w·a"]);
    if (T) cols.push(["dw", "Δw"]);
    head.append(el("th", "#"));
    cols.forEach((c) => head.append(el("th", c[1])));
    head.append(el("th", ""));
    const th = el("thead"); th.append(head); t.append(th);
    const tb = el("tbody"), frag = document.createDocumentFragment();
    for (const r of order) {
      const tr = el("tr");
      tr.append(el("td", other.name + " #" + r.k));
      cols.forEach((c) => tr.append(el("td", fmt(r[c[0]]))));
      tr.append(barCell(r.v, m));
      tr.onmouseenter = () => { hoverEdge = r.edge; draw(); };
      tr.onmouseleave = () => { hoverEdge = null; draw(); };
      frag.append(tr);
    }
    tb.append(frag); t.append(tb); box.append(t);
  }

  // ---- boot ---------------------------------------------------------------------------------------------------------
  new ResizeObserver(() => { if (layers.length) layout(); }).observe($("stage"));
  if (document.fonts) document.fonts.ready.then(() => { if (layers.length) draw(); });

  if (!LIVE) {
    ingest(EMBED);
    setFrame(0, false);
  } else {
    $("followWrap").hidden = false;
    renderFrame(); renderDetail();
    const poll = async () => {
      try {
        const res = await fetch("data?seq=" + lastSeq + "&snap=" + lastSnap, { cache: "no-store" });
        const d = await res.json();
        const curSeq = frames[cur] ? frames[cur].seq : null, before = lastSeq;
        ingest(d);
        $("live").textContent = "live"; $("live").className = "on";
        if (lastSeq !== before) {
          if ($("follow").checked || curSeq == null) setFrame(frames.length - 1);
          else { const k = frames.findIndex((f) => f.seq === curSeq); cur = k < 0 ? 0 : k; updateTimeline(); }
        }
      } catch (err) {
        $("live").textContent = "disconnected"; $("live").className = "off";
      }
      setTimeout(poll, 1000);
    };
    poll();
  }
})();
</script>
</body>
</html>
`
