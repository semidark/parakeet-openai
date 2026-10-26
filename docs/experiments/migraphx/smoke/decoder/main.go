// Standalone ORT 1.25.1 decoder diagnostic, not a deployed server component.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

var inputNames = []string{"encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"}
var outputNames = []string{"outputs", "output_states_1", "output_states_2"}

type worker struct {
	session                                   *ort.AdvancedSession
	frame, state1, state2, logits, out1, out2 *ort.Tensor[float32]
	target, length                            *ort.Tensor[int32]
	values                                    []ort.Value
}

func (w *worker) destroy() {
	if w.session != nil {
		_ = w.session.Destroy()
	}
	for _, value := range w.values {
		_ = value.Destroy()
	}
}

func newWorker(model string, vocab int, opts *ort.SessionOptions) (_ *worker, err error) {
	w := &worker{}
	defer func() {
		if err != nil {
			w.destroy()
		}
	}()
	for _, item := range []struct {
		dest  **ort.Tensor[float32]
		shape ort.Shape
	}{
		{&w.frame, ort.NewShape(1, encoderDim, 1)},
		{&w.state1, ort.NewShape(2, 1, 640)}, {&w.state2, ort.NewShape(2, 1, 640)},
		{&w.logits, ort.NewShape(1, 1, 1, int64(vocab+durations))},
		{&w.out1, ort.NewShape(2, 1, 640)}, {&w.out2, ort.NewShape(2, 1, 640)},
	} {
		*item.dest, err = ort.NewEmptyTensor[float32](item.shape)
		if err != nil {
			return nil, err
		}
		w.values = append(w.values, *item.dest)
	}
	w.target, err = ort.NewEmptyTensor[int32](ort.NewShape(1, 1))
	if err != nil {
		return nil, err
	}
	w.values = append(w.values, w.target)
	w.length, err = ort.NewTensor(ort.NewShape(1), []int32{1})
	if err != nil {
		return nil, err
	}
	w.values = append(w.values, w.length)
	w.session, err = ort.NewAdvancedSession(model, inputNames, outputNames,
		[]ort.ArbitraryTensor{w.frame, w.target, w.length, w.state1, w.state2},
		[]ort.ArbitraryTensor{w.logits, w.out1, w.out2}, opts)
	if err != nil {
		return nil, err
	}
	return w, nil
}

func (w *worker) infer(in input) (output, time.Duration, error) {
	if len(in.Frame) != encoderDim || len(in.State1) != stateSize || len(in.State2) != stateSize {
		return output{}, 0, fmt.Errorf("invalid input sizes")
	}
	copy(w.frame.GetData(), in.Frame)
	w.target.GetData()[0] = in.Target
	copy(w.state1.GetData(), in.State1)
	copy(w.state2.GetData(), in.State2)
	start := time.Now()
	err := w.session.Run()
	elapsed := time.Since(start)
	if err != nil {
		return output{}, elapsed, err
	}
	return output{
		Logits: append([]float32(nil), w.logits.GetData()...),
		State1: append([]float32(nil), w.out1.GetData()...),
		State2: append([]float32(nil), w.out2.GetData()...),
	}, elapsed, nil
}

func inspectModel(model string, vocab int, opts *ort.SessionOptions) error {
	inputs, outputs, err := ort.GetInputOutputInfoWithOptions(model, opts)
	if err != nil {
		return err
	}
	for group, infos := range [][]ort.InputOutputInfo{inputs, outputs} {
		names := inputNames
		shapes := []ort.Shape{{1, encoderDim, 1}, {1, 1}, {1}, {2, 1, 640}, {2, 1, 640}}
		if group == 1 {
			names = outputNames
			shapes = []ort.Shape{{1, 1, 1, int64(vocab + durations)}, {2, 1, 640}, {2, 1, 640}}
		}
		for _, info := range infos {
			fmt.Printf("model %s\n", info.String())
		}
		if group == 0 && len(infos) != len(names) {
			return fmt.Errorf("unexpected model input count")
		}
		for i, name := range names {
			found := false
			for _, info := range infos {
				if info.Name != name {
					continue
				}
				found = true
				expectedType := ort.TensorElementDataType(ort.TensorElementDataTypeFloat)
				if group == 0 && (i == 1 || i == 2) {
					expectedType = ort.TensorElementDataTypeInt32
				}
				if info.OrtValueType != ort.ONNXTypeTensor || info.DataType != expectedType || len(info.Dimensions) != len(shapes[i]) {
					return fmt.Errorf("unexpected interface for %s", name)
				}
				for j, dim := range info.Dimensions {
					if dim >= 0 && dim != shapes[i][j] {
						return fmt.Errorf("%s: dimension %d is %d, expected %d", name, j, dim, shapes[i][j])
					}
				}
			}
			if !found {
				return fmt.Errorf("missing I/O %s", name)
			}
		}
	}
	return nil
}

func buildReference(w *worker, e encoderResult, f *fixture) error {
	// Forced targets exercise recurrent state even when synthetic features emit blank.
	for _, sequence := range []string{"forced", "greedy"} {
		in := zeroInput(f.Blank)
		frame, emitted := 0, 0
		for step := 0; step < f.Steps; step++ {
			if sequence == "forced" {
				frame = step % int(e.Lengths[0])
				targets := []int{f.Blank, 0, 1, f.VocabSize - 2}
				in.Target = int32(targets[step%len(targets)])
			} else if frame >= int(e.Lengths[0]) {
				break
			}
			in.Frame = featureFrame(e, frame)
			o, elapsed, err := w.infer(in)
			if err != nil {
				return err
			}
			d, err := decide(o, f.VocabSize)
			if err != nil {
				return err
			}
			f.Samples = append(f.Samples, sample{sequence, step, frame, in, o, d})
			fmt.Printf("cpu reference sequence=%s step=%d frame=%d token=%d duration=%d native_run=%s\n", sequence, step, frame, d.Token, d.Duration, elapsed)
			if sequence == "forced" {
				in.State1 = append([]float32(nil), o.State1...)
				in.State2 = append([]float32(nil), o.State2...)
			} else {
				advance(d, f.Blank, o, &in, &frame, &emitted)
			}
		}
	}
	return validateFixture(*f, e)
}

func compareSequence(w *worker, f fixture, mode string, atol, rtol float64) error {
	var times []time.Duration
	in := zeroInput(f.Blank)
	frame, emitted := 0, 0
	previous := ""
	for _, ref := range f.Samples {
		if ref.Sequence != previous {
			in = zeroInput(f.Blank)
			frame = 0
			emitted = 0
			previous = ref.Sequence
		}
		if mode == "replay" {
			in = ref.Input
		} else {
			in.Frame = ref.Input.Frame
			if ref.Sequence == "forced" {
				in.Target = ref.Input.Target
			} else if frame != ref.FrameIndex || in.Target != ref.Input.Target {
				return fmt.Errorf("greedy trajectory diverged at step %d", ref.Step)
			}
		}
		o, elapsed, err := w.infer(in)
		if err != nil {
			return fmt.Errorf("%s/%s step=%d: %w", mode, ref.Sequence, ref.Step, err)
		}
		times = append(times, elapsed)
		fmt.Printf("mode=%s sequence=%s step=%d native_run=%s\n", mode, ref.Sequence, ref.Step, elapsed)
		if err := compareOutput(o, ref, f.VocabSize, atol, rtol); err != nil {
			return fmt.Errorf("%s/%s step=%d: %w", mode, ref.Sequence, ref.Step, err)
		}
		if mode == "recurrent" {
			if ref.Sequence == "forced" {
				in.State1 = o.State1
				in.State2 = o.State2
			} else {
				d, _ := decide(o, f.VocabSize)
				advance(d, f.Blank, o, &in, &frame, &emitted)
			}
		}
	}
	// First native Run is reported separately; remaining runs may still include
	// compiler work if the provider specializes on input values. Inspect profiles.
	sorted := append([]time.Duration(nil), times[1:]...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	fmt.Printf("TIMING mode=%s runs=%d first=%s subsequent_p50=%s subsequent_p95=%s (native Run only; not end-to-end)\n", mode, len(times), times[0], sorted[len(sorted)/2], sorted[int(math.Ceil(.95*float64(len(sorted))))-1])
	return nil
}

func run() error {
	model := flag.String("model", "", "FP32 decoder_joint-model.onnx")
	features := flag.String("features", "", "encoder probe gob [1,1024,T] (source audio not inferred)")
	vocabPath := flag.String("vocab", "", "vocab.txt containing <blk>")
	provider := flag.String("provider", "cpu", "cpu writes references; migraphx compares them")
	refs := flag.String("references", "", "decoder fixture gob; CPU refuses overwrite")
	steps := flag.Int("steps", 16, "max steps per CPU sequence (1..1000); GPU uses fixture")
	sessions := flag.Int("sessions", 1, "comparison sessions (1..10)")
	repeats := flag.Int("repeats", 2, "replay/recurrent repetitions per comparison session (1..10)")
	profiles := flag.String("profiles", "", "required fresh directory for ORT profiles")
	cache := flag.String("cache", "", "optional dedicated MIGraphX cache directory")
	mem := flag.Uint64("mem-limit", 6442450944, "MIGraphX arena limit, not total GPU memory")
	atol := flag.Float64("atol", 1e-3, "absolute per-element tolerance")
	rtol := flag.Float64("rtol", 1e-4, "relative per-element tolerance")
	diagnostics := flag.Bool("diagnostics", false, "enable INFO native logs")
	flag.Parse()
	if *model == "" || *features == "" || *vocabPath == "" || *refs == "" || *profiles == "" || (*provider != "cpu" && *provider != "migraphx") {
		return fmt.Errorf("set -model -features -vocab -references -profiles and -provider=cpu|migraphx")
	}
	if *steps < 1 || *steps > 1000 || *sessions < 1 || *sessions > 10 || *repeats < 1 || *repeats > 10 || *mem == 0 || *atol < 0 || *rtol < 0 || math.IsNaN(*atol) || math.IsNaN(*rtol) || math.IsInf(*atol, 0) || math.IsInf(*rtol, 0) {
		return fmt.Errorf("invalid counts, memory limit or tolerances")
	}
	if *provider == "cpu" {
		if _, err := os.Lstat(*refs); !os.IsNotExist(err) {
			return fmt.Errorf("CPU reference path must not exist: %s", *refs)
		}
	}
	vocab, blank, err := vocabulary(*vocabPath)
	if err != nil {
		return err
	}
	var e encoderResult
	if err := readGob(*features, &e); err != nil {
		return err
	}
	if err := validateFeatures(e); err != nil {
		return err
	}
	f := fixture{Version: fixtureVersion, VocabSize: vocab, Blank: blank, Steps: *steps}
	if f.ModelHash, err = fileHash(*model); err != nil {
		return err
	}
	if f.FeaturesHash, err = fileHash(*features); err != nil {
		return err
	}
	if f.VocabHash, err = fileHash(*vocabPath); err != nil {
		return err
	}
	if *provider == "migraphx" {
		var saved fixture
		if err := readGob(*refs, &saved); err != nil {
			return err
		}
		if saved.ModelHash != f.ModelHash || saved.FeaturesHash != f.FeaturesHash || saved.VocabHash != f.VocabHash || saved.VocabSize != vocab || saved.Blank != blank {
			return fmt.Errorf("fixture model/features/vocabulary identity mismatch")
		}
		if err := validateFixture(saved, e); err != nil {
			return err
		}
		f = saved
	}
	lib := os.Getenv("ONNXRUNTIME_LIB")
	if lib == "" {
		return fmt.Errorf("set ONNXRUNTIME_LIB to the exact runtime under test")
	}
	ort.SetSharedLibraryPath(lib)
	if err := ort.InitializeEnvironment(); err != nil {
		return err
	}
	defer ort.DestroyEnvironment()
	if ort.GetVersion() != "1.25.1" {
		return fmt.Errorf("expected ORT 1.25.1, got %s", ort.GetVersion())
	}
	if *diagnostics {
		if err := ort.SetEnvironmentLogLevel(ort.LoggingLevelInfo); err != nil {
			return err
		}
	}
	fmt.Printf("binding=v1.30.1 runtime=%s library=%s provider=%s model_sha256=%s features_sha256=%s vocab=%d blank=%d atol=%g rtol=%g workaround=%q\n", ort.GetVersion(), lib, *provider, f.ModelHash, f.FeaturesHash, vocab, blank, *atol, *rtol, os.Getenv("MIGRAPHX_DISABLE_PASSES"))
	// Metadata inspection uses CPU options, never MIGraphX compiler/session state.
	metadataOpts, err := ort.NewSessionOptions()
	if err != nil {
		return err
	}
	if err := metadataOpts.SetIntraOpNumThreads(4); err != nil {
		metadataOpts.Destroy()
		return err
	}
	err = inspectModel(*model, vocab, metadataOpts)
	metadataOpts.Destroy()
	if err != nil {
		return err
	}
	if err := os.Mkdir(*profiles, 0755); err != nil {
		return fmt.Errorf("profiles require fresh directory: %w", err)
	}
	if *provider == "cpu" {
		*sessions = 1
	}
	for session := 1; session <= *sessions; session++ {
		if err := func() error {
			opts, err := ort.NewSessionOptions()
			if err != nil {
				return err
			}
			defer opts.Destroy()
			if err := opts.SetIntraOpNumThreads(4); err != nil {
				return err
			}
			if *diagnostics {
				if err := opts.SetLogSeverityLevel(ort.LoggingLevelInfo); err != nil {
					return err
				}
			}
			if err := opts.EnableProfiling(filepath.Join(*profiles, fmt.Sprintf("decoder-%s-session%d", *provider, session))); err != nil {
				return err
			}
			if *provider == "migraphx" {
				po := map[string]string{"device_id": "0", "migraphx_mem_limit": strconv.FormatUint(*mem, 10), "migraphx_arena_extend_strategy": "kSameAsRequested"}
				if *cache != "" {
					if err := os.MkdirAll(*cache, 0755); err != nil {
						return err
					}
					po["migraphx_model_cache_dir"] = *cache
				}
				if err := opts.AppendExecutionProvider("MIGraphX", po); err != nil {
					return fmt.Errorf("register MIGraphX: %w", err)
				}
			}
			start := time.Now()
			w, err := newWorker(*model, vocab, opts)
			if err != nil {
				return fmt.Errorf("create decoder session: %w", err)
			}
			defer w.destroy()
			fmt.Printf("session=%d creation=%s\n", session, time.Since(start))
			if *provider == "cpu" {
				if err := buildReference(w, e, &f); err != nil {
					return err
				}
				for _, mode := range []string{"replay", "recurrent"} {
					if err := compareSequence(w, f, mode, *atol, *rtol); err != nil {
						return fmt.Errorf("CPU self-check: %w", err)
					}
				}
				return nil
			}
			for repeat := 1; repeat <= *repeats; repeat++ {
				for _, mode := range []string{"replay", "recurrent"} {
					fmt.Printf("session=%d repeat=%d mode=%s\n", session, repeat, mode)
					if err := compareSequence(w, f, mode, *atol, *rtol); err != nil {
						return err
					}
				}
			}
			return nil
		}(); err != nil {
			return err
		}
	}
	if *provider == "cpu" {
		if err := writeFixture(*refs, f); err != nil {
			return err
		}
		fmt.Printf("CPU FIXTURE: PASS samples=%d path=%s\n", len(f.Samples), *refs)
	} else {
		fmt.Printf("DECODER NUMERICAL/DECISION CHECKS: PASS samples=%d sessions=%d repeats=%d modes=2 (GPU coverage NOT established until profiles are checked)\n", len(f.Samples), *sessions, *repeats)
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
