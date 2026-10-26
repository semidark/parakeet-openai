// Persistent two-process benchmark only. Not part of the application/server.
package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"syscall"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

type request struct {
	ID             int
	Quit, Validate bool
	Features       artifact
}

type response struct {
	ID                                             int
	Error                                          string
	Ready                                          bool
	Role, Provider, Workaround, ModelHash, Runtime string
	SessionCreationNS, StartupNS                   int64
	Features                                       artifact
	Text                                           string
	Decisions                                      []traceDecision
	Trace                                          []traceStep
	EndFrame                                       int
	NativeNS, FirstNativeNS                        int64
	NativeRuns                                     int
}

type measurement struct {
	Provider, Phase, Fixture                               string
	Sequence, Samples, Frames, EncodedFrames, DecodeSteps  int
	TotalNS, PreprocessNS, EncoderRPCNS, DecoderRPCNS      int64
	EncoderNativeNS, DecoderNativeNS, DecoderFirstNativeNS int64
	Text                                                   string
	Decisions                                              []traceDecision
}

type fixtureInput struct {
	Name    string
	WAV     []byte
	Hash    string
	Seconds float64
}

func persistentOptions(provider, profile, cache string) (*ort.SessionOptions, error) {
	o, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	fail := func(e error) (*ort.SessionOptions, error) { o.Destroy(); return nil, e }
	if err = o.SetIntraOpNumThreads(4); err != nil {
		return fail(err)
	}
	if err = o.SetLogSeverityLevel(ort.LoggingLevelWarning); err != nil {
		return fail(err)
	}
	if profile != "" {
		if err = os.Mkdir(profile, 0755); err != nil {
			return fail(err)
		}
		if err = o.EnableProfiling(filepath.Join(profile, "persistent-"+provider)); err != nil {
			return fail(err)
		}
	}
	if provider == "migraphx" {
		if cache == "" {
			return fail(fmt.Errorf("dedicated cache required"))
		}
		po := map[string]string{"device_id": "0", "migraphx_mem_limit": "6442450944", "migraphx_arena_extend_strategy": "kSameAsRequested", "migraphx_model_cache_dir": cache}
		if err = o.AppendExecutionProvider("MIGraphX", po); err != nil {
			return fail(err)
		}
	}
	return o, nil
}

func encoderRequest(s *ort.DynamicAdvancedSession, req request, modelHash string) (response, error) {
	r := response{ID: req.ID}
	mel := req.Features
	if err := validateArtifact(mel, "mel"); err != nil {
		return r, err
	}
	audio, err := ort.NewTensor(ort.Shape(mel.Shape), mel.Data)
	if err != nil {
		return r, err
	}
	defer audio.Destroy()
	length, err := ort.NewTensor(ort.NewShape(1), mel.Lengths)
	if err != nil {
		return r, err
	}
	defer length.Destroy()
	values := make([]ort.Value, 2)
	defer func() {
		for _, v := range values {
			if v != nil {
				v.Destroy()
			}
		}
	}()
	start := time.Now()
	err = s.Run([]ort.Value{audio, length}, values)
	r.NativeNS = time.Since(start).Nanoseconds()
	if err != nil {
		return r, err
	}
	f, ok := values[0].(*ort.Tensor[float32])
	if !ok {
		return r, fmt.Errorf("bad encoder output type")
	}
	l, ok := values[1].(*ort.Tensor[int64])
	if !ok {
		return r, fmt.Errorf("bad encoder length type")
	}
	r.Features = artifact{Version: 1, Kind: "encoder", InputHash: mel.InputHash, ModelHash: modelHash, Shape: append([]int64(nil), f.GetShape()...), Data: append([]float32(nil), f.GetData()...), Lengths: append([]int64(nil), l.GetData()...)}
	if err := validateArtifact(r.Features, "encoder"); err != nil {
		return r, err
	}
	r.NativeRuns = 1
	r.FirstNativeNS = r.NativeNS
	return r, nil
}

// Benchmark path uses reusable tensors and only retains recurrent state on
// nonblank output, matching the application. Full outputs are copied only in
// validation mode; timing mode returns just the transcript and decisions.
func decoderRequest(w *worker, req request, tokens []string, blank int) (response, error) {
	r := response{ID: req.ID}
	f := req.Features
	if err := validateArtifact(f, "encoder"); err != nil {
		return r, err
	}
	clear(w.state1.GetData())
	clear(w.state2.GetData())
	frame, emitted, target := 0, 0, blank
	encoded := int(f.Lengths[0])
	for frame < encoded {
		if r.NativeRuns >= encoded*10 {
			return r, fmt.Errorf("decode bound without completion")
		}
		for d := 0; d < encoderDim; d++ {
			w.frame.GetData()[d] = f.Data[d*encoded+frame]
		}
		w.target.GetData()[0] = int32(target)
		start := time.Now()
		err := w.session.Run()
		elapsed := time.Since(start).Nanoseconds()
		if err != nil {
			return r, err
		}
		if r.NativeRuns == 0 {
			r.FirstNativeNS = elapsed
		}
		r.NativeNS += elapsed
		r.NativeRuns++
		logits := w.logits.GetData()
		d := decision{argmax(logits[:len(tokens)]), argmax(logits[len(tokens):])}
		r.Decisions = append(r.Decisions, traceDecision{frame, d.Token, d.Duration})
		if req.Validate {
			o := output{append([]float32(nil), logits...), append([]float32(nil), w.out1.GetData()...), append([]float32(nil), w.out2.GetData()...)}
			actual, err := decide(o, len(tokens))
			if err != nil {
				return r, err
			}
			if actual != d {
				return r, fmt.Errorf("invalid decision")
			}
			r.Trace = append(r.Trace, traceStep{frame, d, o})
		}
		if d.Token != blank {
			target = d.Token
			copy(w.state1.GetData(), w.out1.GetData())
			copy(w.state2.GetData(), w.out2.GetData())
			emitted++
		}
		if d.Duration > 0 {
			frame += d.Duration
			emitted = 0
		} else if d.Token == blank || emitted >= 10 {
			frame++
			emitted = 0
		}
	}
	steps := make([]traceStep, len(r.Decisions))
	for i, d := range r.Decisions {
		steps[i] = traceStep{Frame: d.Frame, Decision: decision{d.Token, d.Duration}}
	}
	r.Text = transcript(steps, tokens, blank)
	r.EndFrame = frame
	if r.Text == "" {
		return r, fmt.Errorf("empty speech transcript")
	}
	return r, nil
}

func serve(role, provider, socket, profile, cache, models string) error {
	start := time.Now()
	if role != "encoder" && role != "decoder" {
		return fmt.Errorf("invalid worker role")
	}
	workaround, set := os.LookupEnv("MIGRAPHX_DISABLE_PASSES")
	if role == "decoder" && set {
		return fmt.Errorf("decoder pass override must be unset before startup")
	}
	if role == "encoder" && provider == "migraphx" && workaround != "simplify_reshapes" {
		return fmt.Errorf("encoder needs exact workaround")
	}
	if _, err := os.Lstat(socket); !os.IsNotExist(err) {
		return fmt.Errorf("fresh socket required")
	}
	listener, err := net.Listen("unix", socket)
	if err != nil {
		return err
	}
	defer listener.Close()
	defer os.Remove(socket)
	conn, err := listener.Accept()
	if err != nil {
		return err
	}
	defer conn.Close()
	writer, reader := gob.NewEncoder(conn), gob.NewDecoder(conn)
	lib := os.Getenv("ONNXRUNTIME_LIB")
	if lib == "" {
		return fmt.Errorf("exact library required")
	}
	ort.SetSharedLibraryPath(lib)
	if err = ort.InitializeEnvironment(); err != nil {
		return err
	}
	defer ort.DestroyEnvironment()
	if ort.GetVersion() != "1.25.1" {
		return fmt.Errorf("expected ORT 1.25.1")
	}
	if err = ort.SetEnvironmentLogLevel(ort.LoggingLevelWarning); err != nil {
		return err
	}
	opts, err := persistentOptions(provider, profile, cache)
	if err != nil {
		return err
	}
	defer opts.Destroy()
	model := "encoder-model.onnx"
	if role == "decoder" {
		model = "decoder_joint-model.onnx"
	}
	modelHash, err := fileHash(filepath.Join(models, model))
	if err != nil {
		return err
	}
	var encoder *ort.DynamicAdvancedSession
	var decoder *worker
	var tokens []string
	var blank int
	created := time.Now()
	if role == "encoder" {
		encoder, err = ort.NewDynamicAdvancedSession(filepath.Join(models, model), []string{"audio_signal", "length"}, []string{"outputs", "encoded_lengths"}, opts)
		if err != nil {
			return err
		}
		defer encoder.Destroy()
	} else {
		tokens, blank, err = tokenStrings(filepath.Join(models, "vocab.txt"))
		if err != nil {
			return err
		}
		decoder, err = newWorker(filepath.Join(models, model), len(tokens), opts)
		if err != nil {
			return err
		}
		defer decoder.destroy()
	}
	ready := response{Ready: true, Role: role, Provider: provider, Workaround: workaround, ModelHash: modelHash, Runtime: ort.GetVersion(), SessionCreationNS: time.Since(created).Nanoseconds(), StartupNS: time.Since(start).Nanoseconds()}
	if err = writer.Encode(ready); err != nil {
		return err
	}
	for {
		var req request
		if err = reader.Decode(&req); err == io.EOF {
			return nil
		} else if err != nil {
			return err
		}
		if req.Quit {
			return writer.Encode(response{ID: req.ID})
		}
		var r response
		if role == "encoder" {
			r, err = encoderRequest(encoder, req, modelHash)
		} else {
			r, err = decoderRequest(decoder, req, tokens, blank)
		}
		if err != nil {
			r.Error = err.Error()
		}
		if e := writer.Encode(r); e != nil {
			return e
		}
		if err != nil {
			return err
		}
	}
}

type client struct {
	cmd           *exec.Cmd
	conn          net.Conn
	writer        *gob.Encoder
	reader        *gob.Decoder
	ready         response
	id            int
	log           *os.File
	HostStartupNS int64
	stopped       bool
}

var cleared = []string{"MIGRAPHX_DISABLE_PASSES", "ORT_MIGRAPHX_MODEL_CACHE_PATH", "ORT_MIGRAPHX_CACHE_PATH", "ORT_MIGRAPHX_DUMP_MODEL_OPS", "ORT_MIGRAPHX_FP16_ENABLE", "ORT_MIGRAPHX_BF16_ENABLE", "ORT_MIGRAPHX_FP8_ENABLE", "ORT_MIGRAPHX_INT8_ENABLE", "ORT_MIGRAPHX_EXHAUSTIVE_TUNE"}

func workerEnvironment(role, provider string) []string {
	var env []string
	for _, item := range os.Environ() {
		name := strings.SplitN(item, "=", 2)[0]
		remove := false
		for _, bad := range cleared {
			if name == bad {
				remove = true
			}
		}
		if !remove {
			env = append(env, item)
		}
	}
	if role == "encoder" && provider == "migraphx" {
		env = append(env, "MIGRAPHX_DISABLE_PASSES=simplify_reshapes")
	}
	return env
}

func startClient(root, label, role, provider, models string, profile bool) (c *client, err error) {
	c = &client{}
	socket := filepath.Join(root, label+"-"+role+".sock")
	logPath := filepath.Join(root, label+"-"+role+".log")
	c.log, err = os.OpenFile(logPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		return nil, err
	}
	binary, err := os.Executable()
	if err != nil {
		c.log.Close()
		return nil, err
	}
	args := []string{"-role", role, "-provider", provider, "-socket", socket, "-models", models, "-cache", filepath.Join(root, role+"-cache")}
	if profile {
		args = append(args, "-profiles", filepath.Join(root, label+"-"+role+"-profiles"))
	}
	c.cmd = exec.Command(binary, args...)
	c.cmd.Env = workerEnvironment(role, provider)
	c.cmd.Stderr = c.log
	c.cmd.Stdout = c.log
	// Workers die with the diagnostic coordinator; no production processes touched.
	c.cmd.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGTERM}
	start := time.Now()
	if err = c.cmd.Start(); err != nil {
		c.log.Close()
		return nil, err
	}
	defer func() {
		if err != nil {
			c.abort()
		}
	}()
	deadline := time.Now().Add(180 * time.Second)
	for time.Now().Before(deadline) {
		c.conn, err = net.DialTimeout("unix", socket, time.Second)
		if err == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err != nil {
		return c, fmt.Errorf("worker socket: %w", err)
	}
	c.reader = gob.NewDecoder(c.conn)
	c.writer = gob.NewEncoder(c.conn)
	c.conn.SetDeadline(deadline)
	if err = c.reader.Decode(&c.ready); err != nil {
		return c, err
	}
	if !c.ready.Ready || c.ready.Role != role || c.ready.Provider != provider || c.ready.Runtime != "1.25.1" {
		return c, fmt.Errorf("invalid readiness handshake")
	}
	c.HostStartupNS = time.Since(start).Nanoseconds()
	return c, nil
}

func (c *client) call(req request) (response, error) {
	c.id++
	req.ID = c.id
	c.conn.SetDeadline(time.Now().Add(1200 * time.Second))
	if err := c.writer.Encode(req); err != nil {
		return response{}, err
	}
	var r response
	if err := c.reader.Decode(&r); err != nil {
		return r, err
	}
	if r.ID != req.ID {
		return r, fmt.Errorf("response ID mismatch")
	}
	if r.Error != "" {
		return r, fmt.Errorf("worker: %s", r.Error)
	}
	return r, nil
}

func (c *client) abort() {
	if c == nil || c.stopped {
		return
	}
	c.stopped = true
	if c.conn != nil {
		c.conn.Close()
	}
	if c.cmd != nil && c.cmd.Process != nil {
		c.cmd.Process.Kill()
		c.cmd.Wait()
	}
	if c.log != nil {
		c.log.Close()
	}
}
func (c *client) stop() error {
	if c == nil || c.stopped {
		return nil
	}
	_, err := c.call(request{Quit: true})
	if err != nil {
		c.abort()
		return err
	}
	c.conn.Close()
	done := make(chan error, 1)
	go func() { done <- c.cmd.Wait() }()
	select {
	case err = <-done:
	case <-time.After(30 * time.Second):
		c.cmd.Process.Kill()
		err = fmt.Errorf("worker teardown timeout")
		<-done
	}
	c.stopped = true
	c.log.Close()
	return err
}

func preprocess(f fixtureInput) (artifact, int, error) {
	samples, err := parseWAV(f.WAV)
	if err != nil {
		return artifact{}, 0, err
	}
	rows := NewMelFilterbank(128, 16000).Extract(samples)
	if len(rows) == 0 {
		return artifact{}, 0, fmt.Errorf("empty mel")
	}
	a := artifact{Version: 1, Kind: "mel", InputHash: f.Hash, Shape: []int64{1, 128, int64(len(rows))}, Lengths: []int64{int64(len(rows))}, Data: make([]float32, len(rows)*128)}
	for i, row := range rows {
		for m, v := range row {
			a.Data[m*len(rows)+i] = v
		}
	}
	return a, len(samples), validateArtifact(a, "mel")
}

func pipeline(enc, dec *client, f fixtureInput, provider, phase string, sequence int, validate bool) (measurement, response, response, error) {
	m := measurement{Provider: provider, Phase: phase, Fixture: f.Name, Sequence: sequence}
	start := time.Now()
	mel, samples, err := preprocess(f)
	m.PreprocessNS = time.Since(start).Nanoseconds()
	if err != nil {
		return m, response{}, response{}, err
	}
	m.Samples = samples
	m.Frames = int(mel.Shape[2])
	encStart := time.Now()
	e, err := enc.call(request{Features: mel, Validate: validate})
	m.EncoderRPCNS = time.Since(encStart).Nanoseconds()
	if err != nil {
		return m, e, response{}, err
	}
	decStart := time.Now()
	d, err := dec.call(request{Features: e.Features, Validate: validate})
	m.DecoderRPCNS = time.Since(decStart).Nanoseconds()
	m.TotalNS = time.Since(start).Nanoseconds()
	if err != nil {
		return m, e, d, err
	}
	m.EncoderNativeNS = e.NativeNS
	m.DecoderNativeNS = d.NativeNS
	m.DecoderFirstNativeNS = d.FirstNativeNS
	m.EncodedFrames = int(e.Features.Lengths[0])
	m.DecodeSteps = d.NativeRuns
	m.Text = d.Text
	m.Decisions = d.Decisions
	return m, e, d, nil
}

func sameTrajectory(got, ref measurement) error {
	if got.Frames != ref.Frames || got.EncodedFrames != ref.EncodedFrames || got.DecodeSteps != ref.DecodeSteps || got.Text != ref.Text || !reflect.DeepEqual(got.Decisions, ref.Decisions) {
		return fmt.Errorf("full transcript/trajectory mismatch for %s", got.Fixture)
	}
	return nil
}

type validation struct {
	Measurement      measurement
	Encoder, Decoder response
}
type startup struct {
	Label, Role, Provider string
	HostNS                int64
	Ready                 response
}
type summary struct {
	Provider, Schedule, Fixture                                       string
	Count                                                             int
	TotalP50MS, TotalP95MS, TotalMinMS, TotalMaxMS                    float64
	EncoderNativeP50MS, DecoderNativeP50MS, PreprocessP50MS, IPCP50MS float64
	RealTimeFactorP50, RealTimeFactorP95                              float64
}

func percentile(values []int64, p float64) int64 {
	if len(values) == 0 {
		return 0
	}
	v := append([]int64(nil), values...)
	sort.Slice(v, func(i, j int) bool { return v[i] < v[j] })
	idx := int(math.Ceil(p*float64(len(v)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(v) {
		idx = len(v) - 1
	}
	return v[idx]
}

func summarize(all []measurement, fixtures []fixtureInput) []summary {
	var result []summary
	for _, provider := range []string{"cpu", "migraphx"} {
		for _, schedule := range []string{"steady", "mixed"} {
			for _, f := range fixtures {
				var total, enc, dec, prep, ipc []int64
				for _, m := range all {
					if m.Provider == provider && m.Phase == schedule && m.Fixture == f.Name {
						total = append(total, m.TotalNS)
						enc = append(enc, m.EncoderNativeNS)
						dec = append(dec, m.DecoderNativeNS)
						prep = append(prep, m.PreprocessNS)
						ipc = append(ipc, m.TotalNS-m.PreprocessNS-m.EncoderNativeNS-m.DecoderNativeNS)
					}
				}
				if len(total) == 0 {
					continue
				}
				s := summary{Provider: provider, Schedule: schedule, Fixture: f.Name, Count: len(total), TotalP50MS: float64(percentile(total, .5)) / 1e6, TotalP95MS: float64(percentile(total, .95)) / 1e6, TotalMinMS: float64(percentile(total, 0)) / 1e6, TotalMaxMS: float64(percentile(total, 1)) / 1e6, EncoderNativeP50MS: float64(percentile(enc, .5)) / 1e6, DecoderNativeP50MS: float64(percentile(dec, .5)) / 1e6, PreprocessP50MS: float64(percentile(prep, .5)) / 1e6, IPCP50MS: float64(percentile(ipc, .5)) / 1e6}
				s.RealTimeFactorP50 = s.TotalP50MS / (f.Seconds * 1000)
				s.RealTimeFactorP95 = s.TotalP95MS / (f.Seconds * 1000)
				result = append(result, s)
			}
		}
	}
	return result
}

func checkProfiles(root, label, role, provider string, runs int) error {
	paths, err := filepath.Glob(filepath.Join(root, label+"-"+role+"-profiles", "*.json"))
	if err != nil {
		return err
	}
	if len(paths) != 1 {
		return fmt.Errorf("expected exactly one %s profile", role)
	}
	events, err := os.ReadFile(paths[0])
	if err != nil {
		return err
	}
	return strictProfile(events, provider, runs)
}

func coordinate(root, models string, repeats, mixed int) error {
	if repeats < 20 || mixed < 10 {
		return fmt.Errorf("at least 20 steady and 10 mixed requests per fixture required")
	}
	if _, err := os.Lstat(filepath.Join(root, "started.json")); !os.IsNotExist(err) {
		return fmt.Errorf("one-shot run already started")
	}
	if err := writeExclusive(filepath.Join(root, "started.json"), map[string]any{"utc": time.Now().UTC(), "timed_repeats": repeats, "mixed_repeats": mixed, "timed_profiling": false, "threads_per_worker": 4, "mode": "serial two persistent workers; not production/HTTP/concurrency"}, true); err != nil {
		return err
	}
	var fixtures []fixtureInput
	for _, seconds := range []int{2, 6, 10} {
		name := fmt.Sprintf("audio-%ds.wav", seconds)
		path := filepath.Join(root, name)
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		hash, err := fileHash(path)
		if err != nil {
			return err
		}
		fixtures = append(fixtures, fixtureInput{name, b, hash, float64(seconds)})
	}
	baselines := map[string]validation{}
	tokens, blank, err := tokenStrings(filepath.Join(models, "vocab.txt"))
	if err != nil {
		return err
	}
	var all []measurement
	var startups []startup
	maxima := report{MaxAbs: map[string]float64{}}
	var validatedCaches map[string]string
	var trustedSix trace
	if err := readGob(filepath.Join(filepath.Dir(root), "realaudio-v1", "cpu-trace.gob"), &trustedSix); err != nil {
		return err
	}
	for _, provider := range []string{"cpu", "migraphx"} {
		for _, phase := range []string{"validation", "timing"} {
			label := provider + "-" + phase
			fmt.Printf("START %s\n", label)
			enc, err := startClient(root, label, "encoder", provider, models, phase == "validation")
			if err != nil {
				return err
			}
			// The abort defers are idempotent after normal shutdown.
			defer enc.abort()
			dec, err := startClient(root, label, "decoder", provider, models, phase == "validation")
			if err != nil {
				return err
			}
			defer dec.abort()
			for _, c := range []*client{enc, dec} {
				startups = append(startups, startup{label, c.ready.Role, provider, c.HostStartupNS, c.ready})
			}
			encoderRuns, decoderRuns := 0, 0
			do := func(f fixtureInput, kind string, seq int) (validation, error) {
				if kind != "steady" && kind != "mixed" {
					fmt.Printf("REQUEST %s phase=%s fixture=%s seq=%d begin\n", label, kind, f.Name, seq)
				}
				m, e, d, err := pipeline(enc, dec, f, provider, kind, seq, phase == "validation")
				if err != nil {
					return validation{}, err
				}
				encoderRuns++
				decoderRuns += d.NativeRuns
				v := validation{m, e, d}
				if phase == "validation" {
					t := trace{Version: 1, EncodedLength: m.EncodedFrames, EndFrame: d.EndFrame, Text: d.Text, Steps: d.Trace}
					if err := validateTrace(t, e.Features, tokens, blank); err != nil {
						return v, err
					}
					if provider == "cpu" && f.Seconds == 6 {
						if m.Text != trustedSix.Text || d.EndFrame != trustedSix.EndFrame || len(d.Trace) != len(trustedSix.Steps) {
							return v, fmt.Errorf("persistent decoder differs from validated six-second baseline")
						}
						for i, step := range d.Trace {
							if err := compareStep(step, trustedSix.Steps[i], &maxima); err != nil {
								return v, err
							}
						}
					}
				}
				ref, exists := baselines[f.Name]
				if exists {
					if err = sameTrajectory(m, ref.Measurement); err != nil {
						return v, err
					}
					if phase == "validation" {
						if !reflect.DeepEqual(e.Features.Shape, ref.Encoder.Features.Shape) || !reflect.DeepEqual(e.Features.Lengths, ref.Encoder.Features.Lengths) {
							return v, fmt.Errorf("encoder shape mismatch")
						}
						s, err := compareValues("encoder", e.Features.Data, ref.Encoder.Features.Data, 1e-3, 1e-4)
						if err != nil {
							return v, err
						}
						if s.MaxAbs > maxima.MaxAbs["encoder"] {
							maxima.MaxAbs["encoder"] = s.MaxAbs
						}
						if len(d.Trace) != len(ref.Decoder.Trace) {
							return v, fmt.Errorf("trace length mismatch")
						}
						for i, step := range d.Trace {
							if err = compareStep(step, ref.Decoder.Trace[i], &maxima); err != nil {
								return v, err
							}
						}
					}
				} else if provider == "cpu" && phase == "validation" {
					baselines[f.Name] = v
				} else {
					return v, fmt.Errorf("missing CPU reference")
				}
				all = append(all, m)
				if kind != "steady" && kind != "mixed" {
					fmt.Printf("REQUEST %s phase=%s fixture=%s done total_ms=%.3f encoder_native_ms=%.3f steps=%d\n", label, kind, f.Name, float64(m.TotalNS)/1e6, float64(m.EncoderNativeNS)/1e6, m.DecodeSteps)
				}
				return v, nil
			}
			if phase == "validation" {
				for round := 0; round < 2; round++ {
					for _, f := range fixtures {
						v, err := do(f, "validation", round)
						if err != nil {
							return err
						}
						if err = writeExclusive(filepath.Join(root, fmt.Sprintf("%s-%ds-round%d.gob", label, int(f.Seconds), round)), v, false); err != nil {
							return err
						}
					}
				}
			} else {
				for _, f := range fixtures {
					// First request in a new process/shape is reported but excluded from warm stats.
					if _, err = do(f, "first-shape", 0); err != nil {
						return err
					}
					for i := 0; i < 3; i++ {
						if _, err = do(f, "warmup", i); err != nil {
							return err
						}
					}
					for i := 0; i < repeats; i++ {
						if _, err = do(f, "steady", i); err != nil {
							return err
						}
					}
					fmt.Printf("MEASURED %s fixture=%s count=%d\n", label, f.Name, repeats)
				}
				for round := 0; round < mixed; round++ {
					for _, f := range fixtures {
						if _, err = do(f, "mixed", round); err != nil {
							return err
						}
					}
				}
			}
			if err = enc.stop(); err != nil {
				return err
			}
			if err = dec.stop(); err != nil {
				return err
			}
			if phase == "validation" {
				if err = checkProfiles(root, label, "encoder", provider, encoderRuns); err != nil {
					return err
				}
				if err = checkProfiles(root, label, "decoder", provider, decoderRuns); err != nil {
					return err
				}
				fmt.Printf("COVERAGE PASS %s encoder_runs=%d decoder_runs=%d\n", label, encoderRuns, decoderRuns)
			}
			if provider == "migraphx" {
				current, err := cacheHashes(root)
				if err != nil {
					return err
				}
				if phase == "validation" {
					validatedCaches = current
					if err := writeExclusive(filepath.Join(root, "cache-after-validation.json"), current, true); err != nil {
						return err
					}
				} else {
					if !reflect.DeepEqual(validatedCaches, current) {
						return fmt.Errorf("cache changed during timed phase")
					}
					for _, role := range []string{"encoder", "decoder"} {
						data, err := os.ReadFile(filepath.Join(root, label+"-"+role+".log"))
						if err != nil {
							return err
						}
						if strings.Contains(string(data), "Model Compile: Begin") {
							return fmt.Errorf("timed %s unexpectedly compiled", role)
						}
					}
					fmt.Println("TIMED CACHE CHECK: PASS unchanged hashes and no compile markers")
				}
			}
			if err = writeExclusive(filepath.Join(root, label+"-measurements.json"), all, true); err != nil {
				return err
			}
			fmt.Printf("PASS %s\n", label)
		}
	}
	stats := summarize(all, fixtures)
	for _, s := range stats {
		fmt.Printf("RESULT provider=%s schedule=%s fixture=%s n=%d p50_ms=%.3f p95_ms=%.3f rtf_p50=%.4f\n", s.Provider, s.Schedule, s.Fixture, s.Count, s.TotalP50MS, s.TotalP95MS, s.RealTimeFactorP50)
	}
	hashes := map[string]string{}
	for _, f := range fixtures {
		hashes[f.Name] = f.Hash
	}
	return writeExclusive(filepath.Join(root, "result.json"), map[string]any{"status": "PASS", "scope": "serial persistent two-process diagnostic, in-memory WAV through transcript; no HTTP, concurrency or disk reading in timed path", "timing_boundary": "preprocessing + encoder RPC + decoder RPC; correctness checks and result persistence outside clock; full numerical validation/profile separate", "percentile_method": "nearest rank", "sample_counts": "30 steady and 10 mixed per length by default; one execution on one machine, not statistical confidence intervals", "startup": startups, "fixture_sha256": hashes, "maxabs": maxima.MaxAbs, "summary": stats, "measurements": all}, true)
}

func cacheHashes(root string) (map[string]string, error) {
	result := map[string]string{}
	for _, role := range []string{"encoder", "decoder"} {
		files, err := filepath.Glob(filepath.Join(root, role+"-cache", "*.mxr"))
		if err != nil {
			return nil, err
		}
		if len(files) == 0 {
			return nil, fmt.Errorf("%s cache empty", role)
		}
		for _, p := range files {
			hash, err := fileHash(p)
			if err != nil {
				return nil, err
			}
			result[role+"/"+filepath.Base(p)] = hash
		}
	}
	return result, nil
}

func main() {
	role := flag.String("role", "coordinator", "coordinator|encoder|decoder")
	provider := flag.String("provider", "cpu", "cpu|migraphx")
	root := flag.String("root", "/work/persistent-v1", "fresh diagnostic root")
	models := flag.String("models", "/models", "read-only models")
	socket := flag.String("socket", "", "private Unix socket")
	profiles := flag.String("profiles", "", "optional fresh profile directory; timing disabled")
	cache := flag.String("cache", "", "isolated cache")
	repeats := flag.Int("repeats", 30, "steady requests per shape")
	mixed := flag.Int("mixed", 10, "mixed-length cycles")
	flag.Parse()
	var err error
	if *provider != "cpu" && *provider != "migraphx" {
		err = fmt.Errorf("invalid provider")
	} else if *role == "coordinator" {
		err = coordinate(*root, *models, *repeats, *mixed)
	} else {
		err = serve(*role, *provider, *socket, *profiles, *cache, *models)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "PERSISTENT BENCHMARK: FAIL:", err)
		if *role == "coordinator" {
			_ = writeExclusive(filepath.Join(*root, "failure.json"), map[string]string{"error": err.Error()}, true)
		}
		os.Exit(1)
	}
	if *role == "coordinator" {
		fmt.Println("PERSISTENT BENCHMARK: PASS")
	}
}
