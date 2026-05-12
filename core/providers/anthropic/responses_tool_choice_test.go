package anthropic

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// TestConvertAnthropicToolChoiceToBifrost locks in the LOCAL FORK PATCH that
// teaches convertAnthropicToolChoiceToBifrost to inspect the tools list when
// resolving a forced "tool" choice. Pre-patch, every "tool" choice was emitted
// as {type:"function", name:X} regardless of whether the named tool was a
// custom function or an Anthropic server tool (web_search, computer_*, etc.),
// which broke forced server-tool calls -- OpenAI's Responses API rejects
// function-shaped tool_choice against a non-function tool with:
//
//	Tool choice 'function' not found in 'tools' parameter.
//
// If/when upstream Bifrost ships their own fix and we drop the local patch on
// the next merge, this test should still pass against the upstream
// implementation. If it does not, do NOT revert the local patch.
func TestConvertAnthropicToolChoiceToBifrost(t *testing.T) {
	customType := AnthropicToolTypeCustom
	webSearch := AnthropicToolTypeWebSearch20250305
	computer := AnthropicToolTypeComputer20251124
	codeExec := AnthropicToolTypeCodeExecution
	webFetch := AnthropicToolTypeWebFetch20250910

	cases := []struct {
		name       string
		choice     *AnthropicToolChoice
		tools      []AnthropicTool
		wantStr    *string
		wantStruct *schemas.ResponsesToolChoiceStruct
	}{
		{
			name:   "nil choice returns nil",
			choice: nil,
		},
		{
			name:    "auto string passes through",
			choice:  &AnthropicToolChoice{Type: "auto"},
			wantStr: schemas.Ptr("auto"),
		},
		{
			name:    "any string passes through",
			choice:  &AnthropicToolChoice{Type: "any"},
			wantStr: schemas.Ptr("any"),
		},
		{
			name:    "none string passes through",
			choice:  &AnthropicToolChoice{Type: "none"},
			wantStr: schemas.Ptr("none"),
		},
		{
			name:    "unknown string falls back to auto",
			choice:  &AnthropicToolChoice{Type: "weird"},
			wantStr: schemas.Ptr("auto"),
		},
		{
			name:   "tool choice for custom function emits function-shape with name",
			choice: &AnthropicToolChoice{Type: "tool", Name: "get_weather"},
			tools: []AnthropicTool{
				{Name: "get_weather", Type: &customType},
			},
			wantStruct: &schemas.ResponsesToolChoiceStruct{
				Type: schemas.ResponsesToolChoiceTypeFunction,
				Name: schemas.Ptr("get_weather"),
			},
		},
		{
			name:   "tool choice with nil Type treated as custom function",
			choice: &AnthropicToolChoice{Type: "tool", Name: "get_weather"},
			tools: []AnthropicTool{
				{Name: "get_weather", Type: nil},
			},
			wantStruct: &schemas.ResponsesToolChoiceStruct{
				Type: schemas.ResponsesToolChoiceTypeFunction,
				Name: schemas.Ptr("get_weather"),
			},
		},
		{
			// THE REGRESSION CASE. Pre-patch this emitted
			// {type:"function", name:"web_search"} which OpenAI rejects.
			name:   "tool choice for web_search server tool emits web_search type without name",
			choice: &AnthropicToolChoice{Type: "tool", Name: "web_search"},
			tools: []AnthropicTool{
				{Name: "web_search", Type: &webSearch},
			},
			wantStruct: &schemas.ResponsesToolChoiceStruct{
				Type: schemas.ResponsesToolChoiceTypeWebSearch,
			},
		},
		{
			name:   "tool choice for computer server tool emits computer_use_preview type without name",
			choice: &AnthropicToolChoice{Type: "tool", Name: "computer"},
			tools: []AnthropicTool{
				{Name: "computer", Type: &computer},
			},
			wantStruct: &schemas.ResponsesToolChoiceStruct{
				Type: schemas.ResponsesToolChoiceTypeComputerUsePreview,
			},
		},
		{
			name:   "tool choice for code_execution server tool emits code_interpreter type without name",
			choice: &AnthropicToolChoice{Type: "tool", Name: "code_execution"},
			tools: []AnthropicTool{
				{Name: "code_execution", Type: &codeExec},
			},
			wantStruct: &schemas.ResponsesToolChoiceStruct{
				Type: schemas.ResponsesToolChoiceTypeCodeInterpreter,
			},
		},
		{
			// web_fetch has no Responses tool_choice equivalent — fall back to
			// "required" rather than emit an invalid function-shape.
			name:    "tool choice for server tool without Responses equivalent falls back to required",
			choice:  &AnthropicToolChoice{Type: "tool", Name: "web_fetch"},
			tools:   []AnthropicTool{{Name: "web_fetch", Type: &webFetch}},
			wantStr: schemas.Ptr("required"),
		},
		{
			// Inconsistent request (named tool not in tools list) — preserve
			// pre-patch behavior so we don't introduce a new failure mode.
			name:   "tool choice for name not present in tools list falls back to function-shape",
			choice: &AnthropicToolChoice{Type: "tool", Name: "missing"},
			tools:  []AnthropicTool{{Name: "other", Type: &customType}},
			wantStruct: &schemas.ResponsesToolChoiceStruct{
				Type: schemas.ResponsesToolChoiceTypeFunction,
				Name: schemas.Ptr("missing"),
			},
		},
		{
			name:   "tool choice with empty tools list falls back to function-shape",
			choice: &AnthropicToolChoice{Type: "tool", Name: "anything"},
			tools:  nil,
			wantStruct: &schemas.ResponsesToolChoiceStruct{
				Type: schemas.ResponsesToolChoiceTypeFunction,
				Name: schemas.Ptr("anything"),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := convertAnthropicToolChoiceToBifrost(tc.choice, tc.tools)
			if tc.choice == nil {
				if got != nil {
					t.Fatalf("nil choice -> want nil, got %#v", got)
				}
				return
			}
			if got == nil {
				t.Fatalf("non-nil choice -> got nil result")
			}
			if tc.wantStr != nil {
				if got.ResponsesToolChoiceStr == nil || *got.ResponsesToolChoiceStr != *tc.wantStr {
					t.Fatalf("want str %q, got str=%v struct=%v", *tc.wantStr, got.ResponsesToolChoiceStr, got.ResponsesToolChoiceStruct)
				}
				if got.ResponsesToolChoiceStruct != nil {
					t.Fatalf("expected only str result, also got struct=%#v", got.ResponsesToolChoiceStruct)
				}
				return
			}
			if tc.wantStruct != nil {
				if got.ResponsesToolChoiceStruct == nil {
					t.Fatalf("want struct %#v, got nil struct (str=%v)", *tc.wantStruct, got.ResponsesToolChoiceStr)
				}
				gs := got.ResponsesToolChoiceStruct
				if gs.Type != tc.wantStruct.Type {
					t.Errorf("type: got %q want %q", gs.Type, tc.wantStruct.Type)
				}
				if (gs.Name == nil) != (tc.wantStruct.Name == nil) {
					t.Errorf("name nil-ness mismatch: got %v want %v", gs.Name, tc.wantStruct.Name)
				} else if gs.Name != nil && *gs.Name != *tc.wantStruct.Name {
					t.Errorf("name: got %q want %q", *gs.Name, *tc.wantStruct.Name)
				}
			}
		})
	}
}
