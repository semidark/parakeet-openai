// Diagnostic helpers snapshotted from validated real-audio probe.
package main

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
)

const DebugMode = false

type artifact struct {
	Version                    int
	Kind, InputHash, ModelHash string
	Shape                      []int64
	Data                       []float32
	Lengths                    []int64
}

type traceStep struct {
	Frame    int
	Decision decision
	Output   output
}

type trace struct {
	Version                                                         int
	InputHash, EncoderModelHash, FeaturesHash, ModelHash, VocabHash string
	EncodedLength, EndFrame                                         int
	Text                                                            string
	Steps                                                           []traceStep
}

type report struct {
	Stage, Provider, Workaround                                 string
	InputHash, ModelHash, FeaturesHash, VocabHash               string
	Shape                                                       []int64
	Lengths                                                     []int64
	Frames, Samples, Repeats, Runs                              int
	Text                                                        string
	Decisions                                                   []traceDecision
	SessionCreationNS, StageWallNS                              int64
	NativeRunNS                                                 []int64
	NativeFirstNS, NativeSubsequentP50NS, NativeSubsequentP95NS int64
	MaxAbs                                                      map[string]float64
}

type traceDecision struct{ Frame, Token, Duration int }

func writeExclusive(path string, value any, asJSON bool) (err error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	defer func() {
		if e := f.Close(); err == nil {
			err = e
		}
		if err != nil {
			_ = os.Remove(path)
		}
	}()
	if asJSON {
		e := json.NewEncoder(f)
		e.SetIndent("", "  ")
		return e.Encode(value)
	}
	return gob.NewEncoder(f).Encode(value)
}

func validateArtifact(a artifact, kind string) error {
	if a.Version != 1 || a.Kind != kind || len(a.InputHash) != 64 {
		return fmt.Errorf("invalid %s artifact identity", kind)
	}
	if kind == "encoder" {
		if len(a.ModelHash) != 64 {
			return fmt.Errorf("missing encoder identity")
		}
		if err := validateFeatures(encoderResult{a.Shape, a.Data, a.Lengths}); err != nil {
			return err
		}
		// The application uses encoded length as the channel-major stride.
		if a.Lengths[0] != a.Shape[2] {
			return fmt.Errorf("output stride differs from application: shape=%v lengths=%v", a.Shape, a.Lengths)
		}
		return nil
	}
	if len(a.Shape) != 3 || a.Shape[0] != 1 || a.Shape[1] != 128 || a.Shape[2] < 1 || a.Shape[2] > 5000 || len(a.Data) != 128*int(a.Shape[2]) || len(a.Lengths) != 1 || a.Lengths[0] != a.Shape[2] {
		return fmt.Errorf("invalid natural mel shape/length: %v/%v", a.Shape, a.Lengths)
	}
	return finite("mel", a.Data)
}

func tokenStrings(path string) ([]string, int, error) {
	n, blank, err := vocabulary(path)
	if err != nil {
		return nil, 0, err
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	tokens := make([]string, n)
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		fields := strings.Fields(line)
		id, _ := strconv.Atoi(fields[1])
		tokens[id] = strings.ReplaceAll(fields[0], "▁", " ")
	}
	return tokens, blank, nil
}

func transcript(steps []traceStep, tokens []string, blank int) string {
	var parts []string
	for _, s := range steps {
		id := s.Decision.Token
		if id == blank {
			continue
		}
		t := tokens[id]
		if strings.HasPrefix(t, "<") && strings.HasSuffix(t, ">") {
			continue
		}
		parts = append(parts, t)
	}
	return strings.TrimSpace(regexp.MustCompile(`\s{2,}`).ReplaceAllString(strings.Join(parts, ""), " "))
}

func compareStep(got, ref traceStep, r *report) error {
	if got.Frame != ref.Frame || got.Decision != ref.Decision {
		return fmt.Errorf("trajectory mismatch frame/decision got=%d/%+v ref=%d/%+v", got.Frame, got.Decision, ref.Frame, ref.Decision)
	}
	for _, item := range []struct {
		name     string
		got, ref []float32
	}{
		{"logits", got.Output.Logits, ref.Output.Logits}, {"state1", got.Output.State1, ref.Output.State1}, {"state2", got.Output.State2, ref.Output.State2},
	} {
		summary, err := compareValues(item.name, item.got, item.ref, 1e-3, 1e-4)
		if err != nil {
			return err
		}
		if summary.MaxAbs > r.MaxAbs[item.name] {
			r.MaxAbs[item.name] = summary.MaxAbs
		}
	}
	return nil
}

func validateTrace(t trace, features artifact, tokens []string, blank int) error {
	if t.Version != 1 || t.EncodedLength != int(features.Lengths[0]) || len(t.Steps) < 1 || len(t.Steps) > t.EncodedLength*10 {
		return fmt.Errorf("invalid full trace header/count")
	}
	frame, emitted := 0, 0
	in := zeroInput(blank)
	for _, s := range t.Steps {
		if s.Frame != frame || frame >= t.EncodedLength {
			return fmt.Errorf("invalid saved frame trajectory")
		}
		d, err := decide(s.Output, len(tokens))
		if err != nil {
			return err
		}
		if d != s.Decision {
			return fmt.Errorf("invalid saved output decision")
		}
		advance(d, blank, s.Output, &in, &frame, &emitted)
	}
	if frame < t.EncodedLength || frame != t.EndFrame || transcript(t.Steps, tokens, blank) != t.Text {
		return fmt.Errorf("incomplete/inconsistent saved transcript")
	}
	return nil
}
