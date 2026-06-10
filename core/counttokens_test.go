package bifrost

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

func TestAddOpenAICountTokensFallbackForAzure(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Provider: schemas.Azure,
		Model:    "gpt-5.5-cc",
	}

	addOpenAICountTokensFallbackForAzure(req)

	if len(req.Fallbacks) != 1 {
		t.Fatalf("expected one fallback, got %d", len(req.Fallbacks))
	}
	if req.Fallbacks[0].Provider != schemas.OpenAI {
		t.Fatalf("expected OpenAI fallback, got %s", req.Fallbacks[0].Provider)
	}
	if req.Fallbacks[0].Model != "gpt-5.5" {
		t.Fatalf("expected OpenAI fallback model without Azure deployment suffix, got %q", req.Fallbacks[0].Model)
	}
}

func TestAddOpenAICountTokensFallbackForAzurePreservesExplicitFallbacks(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Provider:  schemas.Azure,
		Model:     "gpt-5.5-cc",
		Fallbacks: []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-sonnet-4-6"}},
	}

	addOpenAICountTokensFallbackForAzure(req)

	if len(req.Fallbacks) != 1 || req.Fallbacks[0].Provider != schemas.Anthropic {
		t.Fatalf("expected explicit fallback to be preserved, got %#v", req.Fallbacks)
	}
}

func TestAddOpenAICountTokensFallbackForAzureSkipsNonAzure(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Provider: schemas.OpenAI,
		Model:    "gpt-5.5-cc",
	}

	addOpenAICountTokensFallbackForAzure(req)

	if len(req.Fallbacks) != 0 {
		t.Fatalf("expected no fallback for non-Azure request, got %#v", req.Fallbacks)
	}
}

func TestOpenAICountTokensFallbackModel(t *testing.T) {
	cases := map[string]string{
		"gpt-5.5-cc":      "gpt-5.5",
		"gpt-5.4-mini-cc": "gpt-5.4-mini",
		"gpt-5.5":         "gpt-5.5",
	}

	for input, want := range cases {
		if got := openAICountTokensFallbackModel(input); got != want {
			t.Fatalf("openAICountTokensFallbackModel(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestSanitizeCountTokensRequestStripsResponseOutputControls(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Params: &schemas.ResponsesParameters{
			Include: []string{"reasoning.encrypted_content"},
			Store:   schemas.Ptr(true),
		},
	}

	sanitizeCountTokensRequest(req)

	if len(req.Params.Include) != 0 {
		t.Fatalf("include = %#v, want stripped", req.Params.Include)
	}
	if req.Params.Store != nil {
		t.Fatalf("store = %#v, want stripped", req.Params.Store)
	}
}
