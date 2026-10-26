package main

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestCompareValues(t *testing.T) {
	if _, err := compareValues("x", []float32{1.0001}, []float32{1}, .001, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := compareValues("x", []float32{101}, []float32{100}, 0, .02); err != nil {
		t.Fatal(err)
	}
	for _, got := range [][]float32{nil, {1, 2}, {2}, {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, err := compareValues("x", got, []float32{1}, .001, 0); err == nil {
			t.Fatalf("accepted %v", got)
		}
	}
	if _, err := compareValues("x", []float32{1}, []float32{float32(math.NaN())}, 1, 1); err == nil {
		t.Fatal("accepted non-finite reference")
	}
}

func TestFeatureLayout(t *testing.T) {
	e := encoderResult{Shape: []int64{1, encoderDim, 3}, Lengths: []int64{2}, Data: make([]float32, encoderDim*3)}
	for i := range e.Data {
		e.Data[i] = float32(i)
	}
	if err := validateFeatures(e); err != nil {
		t.Fatal(err)
	}
	f := featureFrame(e, 1)
	for i, v := range f {
		if v != float32(i*3+1) {
			t.Fatalf("channel %d got %g", i, v)
		}
	}
	e.Lengths[0] = 4
	if err := validateFeatures(e); err == nil {
		t.Fatal("accepted length exceeding storage")
	}
}

func TestAdvancePolicy(t *testing.T) {
	o := output{State1: []float32{3}, State2: []float32{4}}
	for _, tc := range []struct {
		name                            string
		d                               decision
		emitted, wantFrame, wantEmitted int
		update                          bool
	}{
		{"blank zero", decision{9, 0}, 0, 1, 0, false},
		{"blank duration", decision{9, 3}, 0, 3, 0, false},
		{"token zero", decision{2, 0}, 0, 0, 1, true},
		{"token duration", decision{2, 2}, 0, 2, 0, true},
		{"token limit", decision{2, 0}, 9, 1, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := input{Target: 9, State1: []float32{1}, State2: []float32{2}}
			frame, emitted := 0, tc.emitted
			advance(tc.d, 9, o, &in, &frame, &emitted)
			if frame != tc.wantFrame || emitted != tc.wantEmitted {
				t.Fatalf("frame/emitted %d/%d", frame, emitted)
			}
			if tc.update {
				if in.Target != 2 || !reflect.DeepEqual(in.State1, o.State1) || !reflect.DeepEqual(in.State2, o.State2) {
					t.Fatal("state feedback missing")
				}
				in.State1[0] = 7
				if o.State1[0] != 3 {
					t.Fatal("feedback aliases output")
				}
			} else if in.Target != 9 || in.State1[0] != 1 || in.State2[0] != 2 {
				t.Fatal("blank updated state")
			}
		})
	}
}

func TestDecisionsAndMismatch(t *testing.T) {
	o := output{Logits: []float32{1, 2, 2, 0, 1, 4, 2, 3}, State1: make([]float32, stateSize), State2: make([]float32, stateSize)}
	d, err := decide(o, 3)
	if err != nil || d != (decision{1, 2}) {
		t.Fatalf("%+v %v", d, err)
	}
	ref := sample{Output: o, Decision: decision{2, 2}}
	if err := compareOutput(o, ref, 3, .001, .0001); err == nil {
		t.Fatal("accepted decision mismatch despite close logits")
	}
	o.Logits = o.Logits[:2]
	if _, err := decide(o, 3); err == nil {
		t.Fatal("accepted wrong output width")
	}
}

func TestVocabulary(t *testing.T) {
	for _, tc := range []struct {
		text  string
		valid bool
	}{
		{"<unk> 0\n<blk> 1\n", true},
		{"<unk> 0\n<blk> 2\n", false},
		{"<unk> 0\n<blk> 0\n", false},
		{"<unk> 0\nword 1\n", false},
		{"<blk> 0\n<blk> 1\n", false},
	} {
		path := filepath.Join(t.TempDir(), "vocab")
		if err := os.WriteFile(path, []byte(tc.text), 0600); err != nil {
			t.Fatal(err)
		}
		count, blank, err := vocabulary(path)
		if (err == nil) != tc.valid {
			t.Fatalf("text=%q err=%v", tc.text, err)
		}
		if tc.valid && (count != 2 || blank != 1) {
			t.Fatal("wrong vocabulary metadata")
		}
	}
}

func TestFixtureRefusesOverwrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fixture.gob")
	f := fixture{Version: fixtureVersion, Steps: 1}
	if err := writeFixture(path, f); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeFixture(path, fixture{}); err == nil {
		t.Fatal("overwrote fixture")
	}
	after, err := os.ReadFile(path)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("existing fixture modified")
	}
	var read fixture
	if err := readGob(path, &read); err != nil || read.Steps != 1 {
		t.Fatalf("%+v %v", read, err)
	}
}
