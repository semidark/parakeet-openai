package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

func strictProfile(data []byte, provider string, expectedRuns int) error {
	var events []struct {
		Name string `json:"name"`
		Args struct {
			Provider string `json:"provider"`
		} `json:"args"`
	}
	if err := json.Unmarshal(data, &events); err != nil {
		return err
	}
	expected := "CPUExecutionProvider"
	if provider == "migraphx" {
		expected = "MIGraphXExecutionProvider"
	}
	runs, kernels := 0, 0
	for _, e := range events {
		if e.Name == "model_run" {
			runs++
		}
		if strings.HasSuffix(e.Name, "_kernel_time") {
			kernels++
			if e.Args.Provider != expected {
				return fmt.Errorf("mixed/missing attribution %q, expected %s", e.Args.Provider, expected)
			}
		}
	}
	if kernels == 0 || runs != expectedRuns {
		return fmt.Errorf("invalid coverage runs=%d expected=%d kernels=%d", runs, expectedRuns, kernels)
	}
	return nil
}
