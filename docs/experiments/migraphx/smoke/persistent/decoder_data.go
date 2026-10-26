// Standalone diagnostic; not imported by the application.
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
)

const (
	encoderDim     = 1024
	stateSize      = 2 * 640
	durations      = 5
	fixtureVersion = 1
)

// Matches the existing encoder probe's gob structure.
type encoderResult struct {
	Shape   []int64
	Data    []float32
	Lengths []int64
}

type input struct {
	Frame          []float32
	Target         int32
	State1, State2 []float32
}

type output struct {
	Logits, State1, State2 []float32
}

type decision struct{ Token, Duration int }
type sample struct {
	Sequence         string
	Step, FrameIndex int
	Input            input
	Output           output
	Decision         decision
}

type fixture struct {
	Version                            int
	ModelHash, FeaturesHash, VocabHash string
	VocabSize, Blank, Steps            int
	Samples                            []sample
}

func fileHash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

func readGob(path string, value any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewDecoder(f).Decode(value)
}

// Refuse overwrites, including existing diagnostic references.
func writeFixture(path string, value fixture) (err error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	defer func() {
		closeErr := f.Close()
		if err == nil {
			err = closeErr
		}
		if err != nil {
			_ = os.Remove(path)
		} // Only the file created by this call.
	}()
	return gob.NewEncoder(f).Encode(value)
}

func vocabulary(path string) (int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()
	ids := make(map[int]bool)
	blank := -1
	s := bufio.NewScanner(f)
	for s.Scan() {
		fields := strings.Fields(s.Text())
		if len(fields) != 2 {
			return 0, 0, fmt.Errorf("invalid vocabulary line %q", s.Text())
		}
		id, err := strconv.Atoi(fields[1])
		if err != nil || id < 0 || id > 1000000 || ids[id] {
			return 0, 0, fmt.Errorf("invalid/duplicate token ID %q", fields[1])
		}
		ids[id] = true
		if fields[0] == "<blk>" {
			if blank >= 0 {
				return 0, 0, fmt.Errorf("multiple blank tokens")
			}
			blank = id
		}
	}
	if err := s.Err(); err != nil {
		return 0, 0, err
	}
	for i := 0; i < len(ids); i++ {
		if !ids[i] {
			return 0, 0, fmt.Errorf("non-contiguous vocabulary IDs")
		}
	}
	if blank < 0 || len(ids) < 2 {
		return 0, 0, fmt.Errorf("vocabulary requires <blk> and at least two tokens")
	}
	return len(ids), blank, nil
}

func validateFeatures(e encoderResult) error {
	if len(e.Shape) != 3 || e.Shape[0] != 1 || e.Shape[1] != encoderDim || e.Shape[2] < 1 || e.Shape[2] > 30000 {
		return fmt.Errorf("expected encoder features [1,1024,T], got %v", e.Shape)
	}
	if len(e.Data) != encoderDim*int(e.Shape[2]) || len(e.Lengths) != 1 || e.Lengths[0] < 1 || e.Lengths[0] > e.Shape[2] {
		return fmt.Errorf("inconsistent encoder feature size/length")
	}
	return finite("encoder features", e.Data)
}

func featureFrame(e encoderResult, frame int) []float32 {
	data := make([]float32, encoderDim)
	for d := range data {
		data[d] = e.Data[d*int(e.Shape[2])+frame]
	}
	return data
}

func finite(name string, data []float32) error {
	for i, value := range data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("%s: non-finite value at %d", name, i)
		}
	}
	return nil
}

type errorsSummary struct{ MaxAbs, MeanAbs float64 }

func compareValues(name string, got, ref []float32, atol, rtol float64) (errorsSummary, error) {
	var result errorsSummary
	if len(got) == 0 || len(got) != len(ref) {
		return result, fmt.Errorf("%s: size mismatch %d/%d", name, len(got), len(ref))
	}
	if err := finite(name+" got", got); err != nil {
		return result, err
	}
	if err := finite(name+" reference", ref); err != nil {
		return result, err
	}
	bad := -1
	for i := range got {
		delta := math.Abs(float64(got[i]) - float64(ref[i]))
		result.MaxAbs = math.Max(result.MaxAbs, delta)
		result.MeanAbs += delta
		if delta > atol+rtol*math.Abs(float64(ref[i])) && bad < 0 {
			bad = i
		}
	}
	result.MeanAbs /= float64(len(got))
	if bad >= 0 {
		return result, fmt.Errorf("%s: tolerance exceeded at %d: got=%g ref=%g maxabs=%g (atol=%g rtol=%g)", name, bad, got[bad], ref[bad], result.MaxAbs, atol, rtol)
	}
	return result, nil
}

func argmax(data []float32) int {
	best := 0
	for i := 1; i < len(data); i++ {
		if data[i] > data[best] {
			best = i
		}
	}
	return best
}

func decide(o output, vocab int) (decision, error) {
	if len(o.Logits) != vocab+durations || len(o.State1) != stateSize || len(o.State2) != stateSize {
		return decision{}, fmt.Errorf("invalid decoder output sizes")
	}
	for name, data := range map[string][]float32{"logits": o.Logits, "state1": o.State1, "state2": o.State2} {
		if err := finite(name, data); err != nil {
			return decision{}, err
		}
	}
	return decision{argmax(o.Logits[:vocab]), argmax(o.Logits[vocab:])}, nil
}

func compareOutput(got output, ref sample, vocab int, atol, rtol float64) error {
	d, err := decide(got, vocab)
	if err != nil {
		return err
	}
	logits, err := compareValues("logits", got.Logits, ref.Output.Logits, atol, rtol)
	if err != nil {
		return err
	}
	s1, err := compareValues("state1", got.State1, ref.Output.State1, atol, rtol)
	if err != nil {
		return err
	}
	s2, err := compareValues("state2", got.State2, ref.Output.State2, atol, rtol)
	if err != nil {
		return err
	}
	if d != ref.Decision {
		return fmt.Errorf("decision mismatch got=%+v ref=%+v", d, ref.Decision)
	}
	fmt.Printf("comparison sequence=%s step=%d token=%d duration=%d logits_maxabs=%.6g logits_meanabs=%.6g state1_maxabs=%.6g state2_maxabs=%.6g\n", ref.Sequence, ref.Step, d.Token, d.Duration, logits.MaxAbs, logits.MeanAbs, s1.MaxAbs, s2.MaxAbs)
	return nil
}

// Exactly the application's blank/state and frame-advance policy.
func advance(d decision, blank int, o output, in *input, frame, emitted *int) {
	if d.Token != blank {
		in.Target = int32(d.Token)
		in.State1 = append([]float32(nil), o.State1...)
		in.State2 = append([]float32(nil), o.State2...)
		(*emitted)++
	}
	if d.Duration > 0 {
		*frame += d.Duration
		*emitted = 0
	} else if d.Token == blank || *emitted >= 10 {
		(*frame)++
		*emitted = 0
	}
}

func zeroInput(blank int) input {
	return input{Target: int32(blank), State1: make([]float32, stateSize), State2: make([]float32, stateSize)}
}

func validateFixture(f fixture, e encoderResult) error {
	if f.Version != fixtureVersion || f.Steps < 1 || f.Steps > 1000 || f.VocabSize < 2 || f.Blank < 0 || f.Blank >= f.VocabSize {
		return fmt.Errorf("invalid fixture header")
	}
	counts := map[string]int{"forced": 0, "greedy": 0}
	for _, s := range f.Samples {
		count, ok := counts[s.Sequence]
		if !ok || s.Step != count || s.FrameIndex < 0 || s.FrameIndex >= int(e.Lengths[0]) || s.Input.Target < 0 || int(s.Input.Target) >= f.VocabSize || len(s.Input.Frame) != encoderDim || len(s.Input.State1) != stateSize || len(s.Input.State2) != stateSize {
			return fmt.Errorf("invalid sample %s/%d", s.Sequence, s.Step)
		}
		if !reflect.DeepEqual(s.Input.Frame, featureFrame(e, s.FrameIndex)) {
			return fmt.Errorf("fixture frame differs from encoder features")
		}
		for name, data := range map[string][]float32{"frame": s.Input.Frame, "state1": s.Input.State1, "state2": s.Input.State2} {
			if err := finite(name, data); err != nil {
				return err
			}
		}
		d, err := decide(s.Output, f.VocabSize)
		if err != nil {
			return err
		}
		if d != s.Decision {
			return fmt.Errorf("invalid saved decision")
		}
		counts[s.Sequence]++
	}
	if counts["forced"] != f.Steps || counts["greedy"] < 1 || counts["greedy"] > f.Steps {
		return fmt.Errorf("incomplete fixture sequences")
	}
	return nil
}
