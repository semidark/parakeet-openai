// Staged real-audio diagnostic. Never loaded by the application/server.
package main

import (
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	ort "github.com/yalue/onnxruntime_go"
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

func prepare(wav, out string, r *report) error {
	data, err := os.ReadFile(wav)
	if err != nil {
		return err
	}
	samples, err := parseWAV(data)
	if err != nil {
		return err
	}
	features := NewMelFilterbank(128, 16000).Extract(samples)
	if len(features) < 1 {
		return fmt.Errorf("no mel frames")
	}
	hash, err := fileHash(wav)
	if err != nil {
		return err
	}
	a := artifact{Version: 1, Kind: "mel", InputHash: hash, Shape: []int64{1, 128, int64(len(features))}, Lengths: []int64{int64(len(features))}}
	a.Data = make([]float32, 128*len(features))
	for f, row := range features {
		if len(row) != 128 {
			return fmt.Errorf("invalid mel row")
		}
		for m, v := range row {
			a.Data[m*len(features)+f] = v
		}
	}
	if err := validateArtifact(a, "mel"); err != nil {
		return err
	}
	r.InputHash, r.Shape, r.Lengths, r.Frames, r.Samples = hash, a.Shape, a.Lengths, len(features), len(samples)
	return writeExclusive(out, a, false)
}

func sessionOptions(provider, profile, cache string, mem uint64) (*ort.SessionOptions, error) {
	o, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	fail := func(e error) (*ort.SessionOptions, error) { o.Destroy(); return nil, e }
	if err := o.SetIntraOpNumThreads(4); err != nil {
		return fail(err)
	}
	if err := o.SetLogSeverityLevel(ort.LoggingLevelInfo); err != nil {
		return fail(err)
	}
	if err := os.Mkdir(profile, 0755); err != nil {
		return fail(fmt.Errorf("fresh profiles required: %w", err))
	}
	if err := o.EnableProfiling(filepath.Join(profile, "realaudio-"+provider)); err != nil {
		return fail(err)
	}
	if provider == "migraphx" {
		po := map[string]string{"device_id": "0", "migraphx_mem_limit": strconv.FormatUint(mem, 10), "migraphx_arena_extend_strategy": "kSameAsRequested"}
		if cache != "" {
			if err := os.MkdirAll(cache, 0755); err != nil {
				return fail(err)
			}
			po["migraphx_model_cache_dir"] = cache
		}
		if err := o.AppendExecutionProvider("MIGraphX", po); err != nil {
			return fail(err)
		}
	}
	return o, nil
}

func encode(model, in, out, reference string, repeats int, opts *ort.SessionOptions, r *report) error {
	var mel artifact
	if err := readGob(in, &mel); err != nil {
		return err
	}
	if err := validateArtifact(mel, "mel"); err != nil {
		return err
	}
	modelHash, err := fileHash(model)
	if err != nil {
		return err
	}
	inHash, err := fileHash(in)
	if err != nil {
		return err
	}
	var ref artifact
	if reference != "" {
		if err := readGob(reference, &ref); err != nil {
			return err
		}
		if err := validateArtifact(ref, "encoder"); err != nil {
			return err
		}
		if ref.InputHash != inHash || ref.ModelHash != modelHash {
			return fmt.Errorf("encoder reference identity mismatch")
		}
	}
	start := time.Now()
	s, err := ort.NewDynamicAdvancedSession(model, []string{"audio_signal", "length"}, []string{"outputs", "encoded_lengths"}, opts)
	if err != nil {
		return err
	}
	defer s.Destroy()
	r.SessionCreationNS = time.Since(start).Nanoseconds()
	audio, err := ort.NewTensor(ort.Shape(mel.Shape), mel.Data)
	if err != nil {
		return err
	}
	defer audio.Destroy()
	length, err := ort.NewTensor(ort.NewShape(1), mel.Lengths)
	if err != nil {
		return err
	}
	defer length.Destroy()
	var first artifact
	for rep := 0; rep < repeats; rep++ {
		var got artifact
		err := func() error {
			outputs := make([]ort.Value, 2)
			defer func() {
				for _, v := range outputs {
					if v != nil {
						v.Destroy()
					}
				}
			}()
			start := time.Now()
			err := s.Run([]ort.Value{audio, length}, outputs)
			elapsed := time.Since(start)
			if err != nil {
				return err
			}
			r.NativeRunNS = append(r.NativeRunNS, elapsed.Nanoseconds())
			f, ok := outputs[0].(*ort.Tensor[float32])
			if !ok {
				return fmt.Errorf("bad encoder feature type %T", outputs[0])
			}
			l, ok := outputs[1].(*ort.Tensor[int64])
			if !ok {
				return fmt.Errorf("bad encoder length type %T", outputs[1])
			}
			got = artifact{1, "encoder", inHash, modelHash, append([]int64(nil), f.GetShape()...), append([]float32(nil), f.GetData()...), append([]int64(nil), l.GetData()...)}
			return validateArtifact(got, "encoder")
		}()
		if err != nil {
			return err
		}
		if rep == 0 {
			first = got
			if reference == "" {
				ref = got
			}
		}
		if !reflect.DeepEqual(got.Shape, ref.Shape) || !reflect.DeepEqual(got.Lengths, ref.Lengths) {
			return fmt.Errorf("encoder shape/length mismatch")
		}
		summary, err := compareValues("encoder", got.Data, ref.Data, 1e-3, 1e-4)
		if err != nil {
			return err
		}
		if summary.MaxAbs > r.MaxAbs["encoder"] {
			r.MaxAbs["encoder"] = summary.MaxAbs
		}
		fmt.Printf("ENCODER repeat=%d shape=%v lengths=%v native_run=%s maxabs=%g\n", rep+1, got.Shape, got.Lengths, time.Duration(r.NativeRunNS[len(r.NativeRunNS)-1]), summary.MaxAbs)
	}
	r.InputHash, r.ModelHash, r.Shape, r.Lengths, r.Frames = inHash, modelHash, first.Shape, first.Lengths, int(mel.Shape[2])
	return writeExclusive(out, first, false)
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

func decode(model, in, out, vocab, reference string, repeats int, opts *ort.SessionOptions, r *report) error {
	var features artifact
	if err := readGob(in, &features); err != nil {
		return err
	}
	if err := validateArtifact(features, "encoder"); err != nil {
		return err
	}
	tokens, blank, err := tokenStrings(vocab)
	if err != nil {
		return err
	}
	modelHash, err := fileHash(model)
	if err != nil {
		return err
	}
	vocabHash, err := fileHash(vocab)
	if err != nil {
		return err
	}
	featuresHash, err := fileHash(in)
	if err != nil {
		return err
	}
	header := trace{Version: 1, InputHash: features.InputHash, EncoderModelHash: features.ModelHash, FeaturesHash: featuresHash, ModelHash: modelHash, VocabHash: vocabHash, EncodedLength: int(features.Lengths[0])}
	var ref trace
	if reference != "" {
		if err := readGob(reference, &ref); err != nil {
			return err
		}
		if ref.InputHash != header.InputHash || ref.EncoderModelHash != header.EncoderModelHash || ref.FeaturesHash != featuresHash || ref.ModelHash != modelHash || ref.VocabHash != vocabHash {
			return fmt.Errorf("decoder reference identity mismatch")
		}
		if err := validateTrace(ref, features, tokens, blank); err != nil {
			return err
		}
	}
	start := time.Now()
	w, err := newWorker(model, len(tokens), opts)
	if err != nil {
		return err
	}
	defer w.destroy()
	r.SessionCreationNS = time.Since(start).Nanoseconds()
	var first trace
	e := encoderResult{features.Shape, features.Data, features.Lengths}
	for rep := 0; rep < repeats; rep++ {
		got := header
		input := zeroInput(blank)
		frame, emitted := 0, 0
		for frame < got.EncodedLength {
			if len(got.Steps) >= got.EncodedLength*10 {
				return fmt.Errorf("TDT bound reached without completion")
			}
			input.Frame = featureFrame(e, frame)
			o, elapsed, err := w.infer(input)
			if err != nil {
				return err
			}
			r.NativeRunNS = append(r.NativeRunNS, elapsed.Nanoseconds())
			d, err := decide(o, len(tokens))
			if err != nil {
				return err
			}
			step := traceStep{frame, d, o}
			idx := len(got.Steps)
			if reference != "" || rep > 0 {
				if idx >= len(ref.Steps) {
					return fmt.Errorf("decoder has extra steps")
				}
				if err := compareStep(step, ref.Steps[idx], r); err != nil {
					return fmt.Errorf("repeat=%d step=%d: %w", rep+1, idx, err)
				}
			}
			got.Steps = append(got.Steps, step)
			advance(d, blank, o, &input, &frame, &emitted)
		}
		got.EndFrame = frame
		got.Text = transcript(got.Steps, tokens, blank)
		if err := validateTrace(got, features, tokens, blank); err != nil {
			return err
		}
		if reference != "" || rep > 0 {
			if len(got.Steps) != len(ref.Steps) || got.Text != ref.Text || got.EndFrame != ref.EndFrame {
				return fmt.Errorf("full decode mismatch")
			}
		}
		if rep == 0 {
			first = got
			if reference == "" {
				ref = got
			}
		}
		fmt.Printf("DECODE repeat=%d complete=true steps=%d encoded_length=%d end_frame=%d text=%q\n", rep+1, len(got.Steps), got.EncodedLength, got.EndFrame, got.Text)
	}
	r.InputHash, r.ModelHash, r.FeaturesHash, r.VocabHash = header.InputHash, modelHash, featuresHash, vocabHash
	r.Text, r.Lengths, r.Shape = first.Text, features.Lengths, features.Shape
	for _, s := range first.Steps {
		r.Decisions = append(r.Decisions, traceDecision{s.Frame, s.Decision.Token, s.Decision.Duration})
	}
	return writeExclusive(out, first, false)
}

func fillTimings(r *report) {
	r.Runs = len(r.NativeRunNS)
	if r.Runs == 0 {
		return
	}
	r.NativeFirstNS = r.NativeRunNS[0]
	if r.Runs < 2 {
		return
	}
	rest := append([]int64(nil), r.NativeRunNS[1:]...)
	sort.Slice(rest, func(i, j int) bool { return rest[i] < rest[j] })
	r.NativeSubsequentP50NS = rest[len(rest)/2]
	r.NativeSubsequentP95NS = rest[(95*len(rest)+99)/100-1]
}

func run() error {
	stage := flag.String("stage", "", "prepare|encode|decode")
	provider := flag.String("provider", "cpu", "cpu|migraphx")
	in := flag.String("input", "", "WAV/mel/features path")
	out := flag.String("output", "", "fresh gob artifact path")
	result := flag.String("report", "", "fresh JSON report path")
	model := flag.String("model", "", "exact FP32 ONNX model")
	vocab := flag.String("vocab", "", "decoder vocabulary")
	reference := flag.String("reference", "", "CPU artifact/full decode trace; required for GPU")
	profile := flag.String("profiles", "", "fresh profile directory")
	cache := flag.String("cache", "", "dedicated stage/compiler cache")
	repeats := flag.Int("repeats", 2, "same-session repetitions, 1..10")
	mem := flag.Uint64("mem-limit", 6442450944, "arena bytes, not a total GPU cap")
	flag.Parse()
	if *stage != "prepare" && *stage != "encode" && *stage != "decode" {
		return fmt.Errorf("invalid stage")
	}
	if *in == "" || *out == "" || *result == "" || (*provider != "cpu" && *provider != "migraphx") || *repeats < 1 || *repeats > 10 || *mem == 0 {
		return fmt.Errorf("invalid arguments")
	}
	for _, p := range []string{*out, *result} {
		if _, err := os.Lstat(p); !os.IsNotExist(err) {
			return fmt.Errorf("fresh output required: %s", p)
		}
	}
	r := report{Stage: *stage, Provider: *provider, Workaround: os.Getenv("MIGRAPHX_DISABLE_PASSES"), Repeats: *repeats, MaxAbs: map[string]float64{}}
	start := time.Now()
	if *stage == "prepare" {
		if err := prepare(*in, *out, &r); err != nil {
			return err
		}
	} else {
		if *model == "" || *profile == "" || (*stage == "decode" && *vocab == "") || (*provider == "migraphx" && *reference == "") {
			return fmt.Errorf("model/profiles/vocab/reference required")
		}
		if *provider == "migraphx" && *stage == "encode" && r.Workaround != "simplify_reshapes" {
			return fmt.Errorf("GPU encoder diagnostic requires exact workaround")
		}
		if *stage == "decode" {
			if _, set := os.LookupEnv("MIGRAPHX_DISABLE_PASSES"); set {
				return fmt.Errorf("decoder requires override UNSET before process start")
			}
		}
		lib := os.Getenv("ONNXRUNTIME_LIB")
		if lib == "" {
			return fmt.Errorf("set exact runtime library")
		}
		ort.SetSharedLibraryPath(lib)
		if err := ort.InitializeEnvironment(); err != nil {
			return err
		}
		defer ort.DestroyEnvironment()
		if ort.GetVersion() != "1.25.1" {
			return fmt.Errorf("expected ORT 1.25.1, got %s", ort.GetVersion())
		}
		if err := ort.SetEnvironmentLogLevel(ort.LoggingLevelInfo); err != nil {
			return err
		}
		fmt.Printf("binding=v1.30.1 runtime=%s library=%s stage=%s provider=%s workaround=%q\n", ort.GetVersion(), lib, *stage, *provider, r.Workaround)
		opts, err := sessionOptions(*provider, *profile, *cache, *mem)
		if err != nil {
			return err
		}
		defer opts.Destroy()
		if *stage == "encode" {
			err = encode(*model, *in, *out, *reference, *repeats, opts, &r)
		} else {
			err = decode(*model, *in, *out, *vocab, *reference, *repeats, opts, &r)
		}
		if err != nil {
			return err
		}
	}
	r.StageWallNS = time.Since(start).Nanoseconds()
	fillTimings(&r)
	if err := writeExclusive(*result, r, true); err != nil {
		return err
	}
	fmt.Printf("REAL-AUDIO STAGE: PASS stage=%s provider=%s runs=%d (provider coverage requires profiles)\n", r.Stage, r.Provider, r.Runs)
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
