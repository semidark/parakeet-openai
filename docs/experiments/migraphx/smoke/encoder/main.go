// Standalone validation tool, not part of the contribution or deployed server.
package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

type result struct {
	Shape   ort.Shape
	Data    []float32
	Lengths []int64
}

func infer(s *ort.DynamicAdvancedSession, frames int) (result, error) {
	data := make([]float32, 128*frames)
	for i := range data {
		data[i] = float32(math.Sin(float64(i)*0.013) + 0.25*math.Cos(float64(i)*0.031))
	}
	audio, err := ort.NewTensor(ort.NewShape(1, 128, int64(frames)), data)
	if err != nil {
		return result{}, err
	}
	defer audio.Destroy()
	length, err := ort.NewTensor(ort.NewShape(1), []int64{int64(frames)})
	if err != nil {
		return result{}, err
	}
	defer length.Destroy()
	outputs := make([]ort.Value, 2)
	defer func() {
		for _, output := range outputs {
			if output != nil {
				output.Destroy()
			}
		}
	}()
	if err := s.Run([]ort.Value{audio, length}, outputs); err != nil {
		return result{}, err
	}
	features, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return result{}, fmt.Errorf("unexpected features type %T", outputs[0])
	}
	encodedLengths, ok := outputs[1].(*ort.Tensor[int64])
	if !ok {
		return result{}, fmt.Errorf("unexpected encoded lengths type %T", outputs[1])
	}
	return result{
		Shape:   append(ort.Shape(nil), features.GetShape()...),
		Data:    append([]float32(nil), features.GetData()...),
		Lengths: append([]int64(nil), encodedLengths.GetData()...),
	}, nil
}

func compare(got, ref result) error {
	if !reflect.DeepEqual(got.Shape, ref.Shape) || !reflect.DeepEqual(got.Lengths, ref.Lengths) || len(got.Data) != len(ref.Data) {
		return fmt.Errorf("shape/length mismatch: got %v/%v, want %v/%v", got.Shape, got.Lengths, ref.Shape, ref.Lengths)
	}
	var maxAbs, sumAbs float64
	for i, value := range got.Data {
		v, r := float64(value), float64(ref.Data[i])
		if math.IsNaN(v) || math.IsInf(v, 0) || math.IsNaN(r) || math.IsInf(r, 0) {
			return fmt.Errorf("non-finite feature at index %d", i)
		}
		delta := math.Abs(v - r)
		maxAbs = math.Max(maxAbs, delta)
		sumAbs += delta
	}
	fmt.Printf("comparison: shape=%v lengths=%v maxabs=%.6g meanabs=%.6g\n", got.Shape, got.Lengths, maxAbs, sumAbs/float64(len(got.Data)))
	if maxAbs > 1e-3 {
		return fmt.Errorf("feature error %.6g exceeds 1e-3", maxAbs)
	}
	return nil
}

func run() error {
	model := flag.String("model", "", "encoder-model.onnx path")
	provider := flag.String("provider", "cpu", "cpu writes references; migraphx compares them")
	framesArg := flag.String("frames", "128,256,128", "frame sequence in one session (tests shape changes)")
	refDir := flag.String("references", "", "reference directory")
	cacheDir := flag.String("cache", "", "optional MIGraphX compiled-model cache")
	profileDir := flag.String("profiles", "", "optional ORT profile directory")
	memLimit := flag.Uint64("mem-limit", 6442450944, "MIGraphX arena limit in bytes (not a total GPU cap)")
	diagnostics := flag.Bool("diagnostics", false, "enable native information-level diagnostics")
	sessionOnly := flag.Bool("session-only", false, "create one session without running inference")
	sessionCount := flag.Int("sessions", 0, "number of sessions (0: CPU=1, MIGraphX=2)")
	flag.Parse()
	if *model == "" || *refDir == "" || (*provider != "cpu" && *provider != "migraphx") {
		return fmt.Errorf("set -model, -references, and -provider=cpu or migraphx")
	}
	if *sessionCount < 0 || *sessionCount > 10 {
		return fmt.Errorf("set -sessions between 0 and 10")
	}
	var frames []int
	for _, text := range strings.Split(*framesArg, ",") {
		n, err := strconv.Atoi(text)
		if err != nil || n < 1 || n > 30000 {
			return fmt.Errorf("invalid frame count %q", text)
		}
		frames = append(frames, n)
	}
	lib := os.Getenv("ONNXRUNTIME_LIB")
	if lib == "" {
		return fmt.Errorf("set ONNXRUNTIME_LIB to the exact library under test")
	}
	ort.SetSharedLibraryPath(lib)
	if err := ort.InitializeEnvironment(); err != nil {
		return err
	}
	defer ort.DestroyEnvironment()
	if *diagnostics {
		if err := ort.SetEnvironmentLogLevel(ort.LoggingLevelInfo); err != nil {
			return err
		}
	}
	if ort.GetVersion() != "1.25.1" {
		return fmt.Errorf("expected runtime 1.25.1, got %s", ort.GetVersion())
	}
	fmt.Printf("binding=v1.30.1 runtime=%s library=%s provider=%s\n", ort.GetVersion(), lib, *provider)
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return err
	}
	defer opts.Destroy()
	if *diagnostics {
		if err := opts.SetLogSeverityLevel(ort.LoggingLevelInfo); err != nil {
			return err
		}
	}
	if err := opts.SetIntraOpNumThreads(4); err != nil {
		return err
	}
	if *profileDir != "" {
		if err := os.MkdirAll(*profileDir, 0755); err != nil {
			return err
		}
		if err := opts.EnableProfiling(filepath.Join(*profileDir, "encoder-"+*provider)); err != nil {
			return err
		}
	}
	if *provider == "migraphx" {
		providerOpts := map[string]string{
			"device_id":                      "0",
			"migraphx_mem_limit":             strconv.FormatUint(*memLimit, 10),
			"migraphx_arena_extend_strategy": "kSameAsRequested",
		}
		if *cacheDir != "" {
			if err := os.MkdirAll(*cacheDir, 0755); err != nil {
				return err
			}
			providerOpts["migraphx_model_cache_dir"] = *cacheDir
		}
		if err := opts.AppendExecutionProvider("MIGraphX", providerOpts); err != nil {
			return fmt.Errorf("register MIGraphX: %w", err)
		}
	}
	if *provider == "cpu" {
		if err := os.MkdirAll(*refDir, 0755); err != nil {
			return err
		}
	}
	// Session recreation permits checking cache reuse separately from warm runs.
	sessions := 1
	if *provider == "migraphx" && !*sessionOnly {
		sessions = 2
	}
	if *sessionCount > 0 && !*sessionOnly {
		sessions = *sessionCount
	}
	fmt.Printf("sessions=%d cache=%q frames=%v\n", sessions, *cacheDir, frames)
	for sessionIdx := 0; sessionIdx < sessions; sessionIdx++ {
		if err := func() error {
			start := time.Now()
			s, err := ort.NewDynamicAdvancedSession(*model, []string{"audio_signal", "length"}, []string{"outputs", "encoded_lengths"}, opts)
			if err != nil {
				return err
			}
			defer s.Destroy()
			fmt.Printf("session=%d creation=%s\n", sessionIdx+1, time.Since(start))
			if *sessionOnly {
				return nil
			}
			for _, n := range frames {
				for rep := 0; rep < 2; rep++ {
					fmt.Printf("session=%d frames=%d run=%d starting\n", sessionIdx+1, n, rep+1)
					start := time.Now()
					got, err := infer(s, n)
					if err != nil {
						return fmt.Errorf("frames=%d run=%d: %w", n, rep+1, err)
					}
					fmt.Printf("session=%d frames=%d run=%d elapsed=%s\n", sessionIdx+1, n, rep+1, time.Since(start))
					path := filepath.Join(*refDir, fmt.Sprintf("encoder-T%d.gob", n))
					if *provider == "cpu" {
						f, err := os.Create(path)
						if err != nil {
							return err
						}
						err = gob.NewEncoder(f).Encode(got)
						closeErr := f.Close()
						if err != nil {
							return err
						}
						if closeErr != nil {
							return closeErr
						}
					} else {
						f, err := os.Open(path)
						if err != nil {
							return err
						}
						var ref result
						err = gob.NewDecoder(f).Decode(&ref)
						f.Close()
						if err != nil {
							return err
						}
						if err := compare(got, ref); err != nil {
							return err
						}
					}
				}
			}
			return nil
		}(); err != nil {
			return err
		}
	}
	if *sessionOnly {
		fmt.Println("SESSION CREATION: PASS (no inference performed)")
	} else {
		fmt.Println("ENCODER PROBE: PASS (provider coverage requires profile inspection)")
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
