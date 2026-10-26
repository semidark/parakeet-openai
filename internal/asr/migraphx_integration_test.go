// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"os"
	"strings"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

// This opt-in test verifies native API compatibility and fail-fast provider
// registration against the standard CPU-only ORT 1.25.1 distribution. It does
// not require a GPU and deliberately does not claim inference coverage.
func TestMIGraphXCPUOnlyRuntime(t *testing.T) {
	lib := os.Getenv("PARAKEET_TEST_ORT125_CPU_LIB")
	if lib == "" {
		t.Skip("set PARAKEET_TEST_ORT125_CPU_LIB to a CPU-only ORT 1.25.1 library")
	}
	if ort.IsInitialized() {
		t.Fatal("test requires an uninitialized runtime")
	}
	ort.SetSharedLibraryPath(lib)
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	defer ort.DestroyEnvironment()
	if got := ort.GetVersion(); got != "1.25.1" {
		t.Fatalf("runtime version = %q, want 1.25.1", got)
	}
	opts, err := buildSessionOptions(GPUConfig{Provider: ProviderMIGraphX})
	if opts != nil {
		defer opts.Destroy()
	}
	if err == nil {
		t.Fatal("CPU-only runtime unexpectedly accepted MIGraphX")
	}
	for _, want := range []string{"enable MIGraphX", "not supported in this build"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q does not contain %q", err, want)
		}
	}
	if opts != nil {
		t.Fatal("failed registration must not return session options")
	}
}
