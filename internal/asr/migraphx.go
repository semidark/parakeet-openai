// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"context"
	"fmt"
	"strconv"
)

// withEncoderRunGate serializes only the shared MIGraphX encoder Run, not
// decoding. A nil gate preserves the CPU/CUDA path. Waiting is cancellable;
// this does not interrupt a native Run that has already started.
func withEncoderRunGate(ctx context.Context, gate chan struct{}, run func() error) error {
	if gate == nil {
		return run()
	}
	select {
	case gate <- struct{}{}:
		defer func() { <-gate }()
	case <-ctx.Done():
		return ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return run()
}

// migraphxProviderOptions keeps option construction independent of the native
// runtime so the configuration contract can be tested without an AMD GPU.
func migraphxProviderOptions(gpu GPUConfig) (map[string]string, error) {
	if gpu.DeviceID < 0 {
		return nil, fmt.Errorf("MIGraphX GPU device index must be non-negative, got %d", gpu.DeviceID)
	}
	opts := map[string]string{
		"device_id":                      strconv.Itoa(gpu.DeviceID),
		"migraphx_arena_extend_strategy": "kSameAsRequested",
	}
	if gpu.MIGraphXMemLimit != 0 {
		opts["migraphx_mem_limit"] = strconv.FormatUint(gpu.MIGraphXMemLimit, 10)
	}
	if gpu.MIGraphXCacheDir != "" {
		opts["migraphx_model_cache_dir"] = gpu.MIGraphXCacheDir
	}
	return opts, nil
}
