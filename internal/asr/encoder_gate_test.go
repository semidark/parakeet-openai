// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
)

func TestEncoderGateSerializesNativeRuns(t *testing.T) {
	gate := make(chan struct{}, 1)
	var wg sync.WaitGroup
	var active, calls atomic.Int32
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := withEncoderRunGate(context.Background(), gate, func() error {
				if n := active.Add(1); n != 1 {
					t.Errorf("%d simultaneous native Runs", n)
				}
				calls.Add(1)
				active.Add(-1)
				return nil
			}); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if calls.Load() != 20 || len(gate) != 0 {
		t.Fatalf("calls=%d gate occupancy=%d", calls.Load(), len(gate))
	}
}

func TestEncoderGateCancellationAndErrorRelease(t *testing.T) {
	gate := make(chan struct{}, 1)
	gate <- struct{}{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	called := false
	err := withEncoderRunGate(ctx, gate, func() error { called = true; return nil })
	if !errors.Is(err, context.Canceled) || called {
		t.Fatalf("canceled wait: error=%v called=%v", err, called)
	}
	<-gate
	sentinel := errors.New("native Run failed")
	err = withEncoderRunGate(context.Background(), gate, func() error { return sentinel })
	if !errors.Is(err, sentinel) || len(gate) != 0 {
		t.Fatalf("failed Run: error=%v gate occupancy=%d", err, len(gate))
	}
	err = withEncoderRunGate(ctx, gate, func() error { called = true; return nil })
	if !errors.Is(err, context.Canceled) || called || len(gate) != 0 {
		t.Fatalf("canceled free gate: error=%v called=%v occupancy=%d", err, called, len(gate))
	}
}

func TestEncoderNilGatePreservesOriginalRun(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	called := false
	if err := withEncoderRunGate(ctx, nil, func() error { called = true; return nil }); err != nil || !called {
		t.Fatalf("nil gate changed original path: error=%v called=%v", err, called)
	}
}
