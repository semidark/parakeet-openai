// Snapshot of the validated standalone decoder worker and metadata inspection.
package main

import (
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
	"time"
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
