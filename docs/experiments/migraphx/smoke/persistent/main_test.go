package main

import (
	"encoding/gob"
	"encoding/json"
	"math"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestNearestRank(t *testing.T) {
	values := make([]int64, 30)
	for i := range values {
		values[i] = int64(30 - i)
	}
	for _, tc := range []struct {
		p    float64
		want int64
	}{{0, 1}, {.5, 15}, {.95, 29}, {1, 30}} {
		if got := percentile(values, tc.p); got != tc.want {
			t.Fatalf("p=%g got=%d want=%d", tc.p, got, tc.want)
		}
	}
	if values[0] != 30 {
		t.Fatal("mutated source")
	}
	if percentile(nil, .5) != 0 {
		t.Fatal("bad empty result")
	}
}

func TestWarmSummaryExclusions(t *testing.T) {
	all := []measurement{
		{Provider: "migraphx", Phase: "first-shape", Fixture: "audio", TotalNS: 999000000},
		{Provider: "migraphx", Phase: "validation", Fixture: "audio", TotalNS: 999000000},
		{Provider: "migraphx", Phase: "warmup", Fixture: "audio", TotalNS: 999000000},
		{Provider: "migraphx", Phase: "steady", Fixture: "audio", TotalNS: 10000000, PreprocessNS: 1000000, EncoderNativeNS: 2000000, DecoderNativeNS: 3000000},
		{Provider: "migraphx", Phase: "mixed", Fixture: "audio", TotalNS: 20000000},
	}
	result := summarize(all, []fixtureInput{{Name: "audio", Seconds: 2}})
	if len(result) != 2 || result[0].Count != 1 || result[0].TotalP50MS != 10 || result[0].IPCP50MS != 4 || result[0].RealTimeFactorP50 != .005 {
		t.Fatalf("bad summaries %+v", result)
	}
	if result[1].Schedule != "mixed" || result[1].TotalP50MS != 20 {
		t.Fatalf("bad mixed %+v", result)
	}
}

func TestWorkerEnvironment(t *testing.T) {
	for _, name := range cleared {
		t.Setenv(name, "inherited-bad-value")
	}
	t.Setenv("ONNXRUNTIME_LIB", "/pinned/lib.so")
	for _, role := range []string{"encoder", "decoder"} {
		for _, provider := range []string{"cpu", "migraphx"} {
			env := workerEnvironment(role, provider)
			values := map[string]string{}
			for _, item := range env {
				p := strings.SplitN(item, "=", 2)
				values[p[0]] = p[1]
			}
			for _, name := range cleared {
				got, exists := values[name]
				if name == "MIGRAPHX_DISABLE_PASSES" && role == "encoder" && provider == "migraphx" {
					if got != "simplify_reshapes" {
						t.Fatal("missing override")
					}
				} else if exists {
					t.Fatalf("%s leaked to %s/%s", name, role, provider)
				}
			}
			if values["ONNXRUNTIME_LIB"] != "/pinned/lib.so" {
				t.Fatal("lost pinned library")
			}
		}
	}
}

func TestStrictProfiles(t *testing.T) {
	for _, tc := range []struct {
		providers []string
		runs      int
		pass      bool
	}{
		{[]string{"MIGraphXExecutionProvider"}, 2, true},
		{[]string{"CPUExecutionProvider"}, 2, false},
		{[]string{"MIGraphXExecutionProvider", "CPUExecutionProvider"}, 2, false},
		{[]string{"MIGraphXExecutionProvider", ""}, 2, false},
		{nil, 2, false}, {[]string{"MIGraphXExecutionProvider"}, 1, false},
	} {
		var events []map[string]any
		for i := 0; i < tc.runs; i++ {
			events = append(events, map[string]any{"name": "model_run"})
		}
		for _, p := range tc.providers {
			events = append(events, map[string]any{"name": "node_kernel_time", "args": map[string]string{"provider": p}})
		}
		b, _ := json.Marshal(events)
		err := strictProfile(b, "migraphx", 2)
		if (err == nil) != tc.pass {
			t.Fatalf("%+v error=%v", tc, err)
		}
	}
}

func TestPersistentProtocol(t *testing.T) {
	a, b := net.Pipe()
	defer a.Close()
	defer b.Close()
	c := &client{conn: a, writer: gob.NewEncoder(a), reader: gob.NewDecoder(a)}
	done := make(chan error, 1)
	go func() {
		reader, writer := gob.NewDecoder(b), gob.NewEncoder(b)
		for i := 1; i <= 3; i++ {
			var req request
			if err := reader.Decode(&req); err != nil {
				done <- err
				return
			}
			if err := writer.Encode(response{ID: req.ID, Text: "speech"}); err != nil {
				done <- err
				return
			}
		}
		done <- nil
	}()
	for i := 1; i <= 3; i++ {
		r, err := c.call(request{})
		if err != nil || r.ID != i || r.Text != "speech" {
			t.Fatalf("bad response %+v %v", r, err)
		}
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestBadProtocolResponses(t *testing.T) {
	for _, r := range []response{{ID: 99}, {ID: 1, Error: "failed"}} {
		a, b := net.Pipe()
		c := &client{conn: a, writer: gob.NewEncoder(a), reader: gob.NewDecoder(a)}
		go func(r response) {
			defer b.Close()
			var req request
			gob.NewDecoder(b).Decode(&req)
			gob.NewEncoder(b).Encode(r)
		}(r)
		if _, err := c.call(request{}); err == nil {
			t.Fatal("accepted bad response")
		}
		a.Close()
	}
}

func TestTrajectoryMismatch(t *testing.T) {
	ref := measurement{Frames: 598, EncodedFrames: 75, DecodeSteps: 1, Text: "speech", Decisions: []traceDecision{{0, 1, 2}}}
	if err := sameTrajectory(ref, ref); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		got := ref
		switch i {
		case 0:
			got.Frames++
		case 1:
			got.EncodedFrames++
		case 2:
			got.DecodeSteps++
		case 3:
			got.Text = "wrong"
		case 4:
			got.Decisions = []traceDecision{{0, 1, 3}}
		}
		if sameTrajectory(got, ref) == nil {
			t.Fatal("accepted mismatch")
		}
	}
}

func TestNaturalLengthsAndExclusiveEvidence(t *testing.T) {
	for _, seconds := range []int{2, 6, 10} {
		samples := make([]float32, seconds*16000)
		for i := range samples {
			samples[i] = float32(math.Sin(float64(i) * .09))
		}
		rows := NewMelFilterbank(128, 16000).Extract(samples)
		if len(rows) != seconds*100-2 {
			t.Fatalf("%ds frames=%d", seconds, len(rows))
		}
	}
	p := filepath.Join(t.TempDir(), "result.json")
	if err := writeExclusive(p, map[string]int{"original": 1}, true); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(p)
	if writeExclusive(p, map[string]int{"changed": 1}, true) == nil {
		t.Fatal("accepted overwrite")
	}
	after, _ := os.ReadFile(p)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("modified evidence")
	}
}
