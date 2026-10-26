// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"reflect"
	"testing"
)

func TestMIGraphXProviderOptions(t *testing.T) {
	cases := []struct {
		name string
		gpu  GPUConfig
		want map[string]string
	}{
		{
			name: "defaults preserve runtime memory and precision defaults",
			want: map[string]string{
				"device_id": "0", "migraphx_arena_extend_strategy": "kSameAsRequested",
			},
		},
		{
			name: "device memory limit and cache are forwarded",
			gpu:  GPUConfig{DeviceID: 1, MIGraphXMemLimit: 6442450944, MIGraphXCacheDir: "/cache/compiled models"},
			want: map[string]string{
				"device_id": "1", "migraphx_arena_extend_strategy": "kSameAsRequested",
				"migraphx_mem_limit": "6442450944", "migraphx_model_cache_dir": "/cache/compiled models",
			},
		},
		{
			name: "memory limit uses unsigned decimal",
			gpu:  GPUConfig{MIGraphXMemLimit: ^uint64(0)},
			want: map[string]string{
				"device_id": "0", "migraphx_arena_extend_strategy": "kSameAsRequested",
				"migraphx_mem_limit": "18446744073709551615",
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := migraphxProviderOptions(tc.gpu)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("options = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestMIGraphXProviderOptionsRejectNegativeDevice(t *testing.T) {
	if _, err := migraphxProviderOptions(GPUConfig{DeviceID: -1}); err == nil {
		t.Fatal("expected a negative device index to be rejected")
	}
}
