package main

import (
	"fmt"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

func run() error {
	path := os.Getenv("ONNXRUNTIME_LIB")
	if path == "" {
		return fmt.Errorf("ONNXRUNTIME_LIB must identify the library under test")
	}
	ort.SetSharedLibraryPath(path)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("initialize runtime: %w", err)
	}
	defer ort.DestroyEnvironment()
	fmt.Printf("Binding: github.com/yalue/onnxruntime_go v1.30.1\nLibrary: %s\nRuntime: %s\n", path, ort.GetVersion())
	if ort.GetVersion() != "1.25.1" {
		return fmt.Errorf("expected runtime 1.25.1, got %s", ort.GetVersion())
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("create session options: %w", err)
	}
	defer opts.Destroy()
	fmt.Println("Native initialization and session-options creation: PASS")
	if len(os.Args) > 1 {
		name := os.Args[1]
		err := opts.AppendExecutionProvider(name, map[string]string{"device_id": "0"})
		fmt.Printf("Generic registration %q: %v\n", name, err)
		if err != nil {
			return fmt.Errorf("generic provider registration: %w", err)
		}
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
