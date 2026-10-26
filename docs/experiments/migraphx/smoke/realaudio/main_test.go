package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestNaturalMelFrames(t *testing.T) {
	samples := make([]float32, 6*16000)
	for i := range samples {
		samples[i] = float32(math.Sin(float64(i) * .09))
	}
	got := NewMelFilterbank(128, 16000).Extract(samples)
	if len(got) != 598 {
		t.Fatalf("natural frames=%d, want 598 (no bucketing/padding)", len(got))
	}
	if len(got[0]) != 128 {
		t.Fatal("wrong mel size")
	}
}

func TestArtifactStrideAndFinite(t *testing.T) {
	a := artifact{Version: 1, Kind: "encoder", InputHash: string(make([]byte, 64)), ModelHash: string(make([]byte, 64)), Shape: []int64{1, 1024, 2}, Data: make([]float32, 2048), Lengths: []int64{2}}
	if err := validateArtifact(a, "encoder"); err != nil {
		t.Fatal(err)
	}
	a.Lengths[0] = 1
	if validateArtifact(a, "encoder") == nil {
		t.Fatal("accepted stride mismatch")
	}
	a.Lengths[0] = 2
	a.Data[0] = float32(math.Inf(1))
	if validateArtifact(a, "encoder") == nil {
		t.Fatal("accepted infinity")
	}
}

func TestCompleteTraceAndTranscript(t *testing.T) {
	features := artifact{Lengths: []int64{2}}
	tokens := []string{" Hello", "<blk>"}
	makeOutput := func(token, duration int) output {
		o := output{Logits: make([]float32, len(tokens)+durations), State1: make([]float32, stateSize), State2: make([]float32, stateSize)}
		o.Logits[token] = 2
		o.Logits[len(tokens)+duration] = 2
		return o
	}
	steps := []traceStep{{0, decision{0, 1}, makeOutput(0, 1)}, {1, decision{1, 1}, makeOutput(1, 1)}}
	ref := trace{Version: 1, EncodedLength: 2, EndFrame: 2, Text: "Hello", Steps: steps}
	if err := validateTrace(ref, features, tokens, 1); err != nil {
		t.Fatal(err)
	}
	ref.Steps = steps[:1]
	ref.EndFrame = 1
	if validateTrace(ref, features, tokens, 1) == nil {
		t.Fatal("accepted truncated transcription")
	}
}

func TestTrajectoryAndNumericalMismatch(t *testing.T) {
	o := output{Logits: []float32{1}, State1: []float32{0}, State2: []float32{0}}
	step := traceStep{0, decision{1, 2}, o}
	r := report{MaxAbs: map[string]float64{}}
	if err := compareStep(step, step, &r); err != nil {
		t.Fatal(err)
	}
	different := step
	different.Frame = 1
	if compareStep(different, step, &r) == nil {
		t.Fatal("accepted frame mismatch")
	}
	different = step
	different.Output.Logits = []float32{2}
	if compareStep(different, step, &r) == nil {
		t.Fatal("accepted logits mismatch")
	}
}

func TestExclusiveOutputs(t *testing.T) {
	p := filepath.Join(t.TempDir(), "artifact.json")
	if err := writeExclusive(p, map[string]int{"original": 1}, true); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if writeExclusive(p, map[string]int{"changed": 2}, true) == nil {
		t.Fatal("accepted overwrite")
	}
	after, _ := os.ReadFile(p)
	if string(before) != string(after) {
		t.Fatal("changed existing evidence")
	}
}

func TestTimingSmallCounts(t *testing.T) {
	for _, times := range [][]int64{nil, {5}, {5, 7}, {5, 9, 3}} {
		r := report{NativeRunNS: times}
		fillTimings(&r)
		if r.Runs != len(times) {
			t.Fatal("bad count")
		}
	}
}
