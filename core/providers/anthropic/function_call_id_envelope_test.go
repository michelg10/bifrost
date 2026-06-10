package anthropic

import (
	"context"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

func TestFunctionCallToolUseIDEnvelopeRoundTrip(t *testing.T) {
	t.Parallel()

	callID := schemas.Ptr("call_12345xyz")
	itemID := schemas.Ptr("fc_12345xyz")

	encoded := encodeFunctionCallToolUseID(callID, itemID)
	if encoded == nil {
		t.Fatal("encoded ID is nil")
	}
	if !strings.HasPrefix(*encoded, *callID+functionCallIDEnvelopeMarker) {
		t.Fatalf("encoded ID = %q, want call ID prefix and marker", *encoded)
	}

	decodedCallID, decodedItemID, ok := decodeFunctionCallToolUseID(encoded)
	if !ok {
		t.Fatal("decode failed")
	}
	if decodedCallID == nil || *decodedCallID != *callID {
		t.Fatalf("decoded call ID = %v, want %q", decodedCallID, *callID)
	}
	if decodedItemID == nil || *decodedItemID != *itemID {
		t.Fatalf("decoded item ID = %v, want %q", decodedItemID, *itemID)
	}
}

func TestFunctionCallToolUseIDEnvelopeSkipsMissingOrNonFunctionItemID(t *testing.T) {
	t.Parallel()

	callID := schemas.Ptr("call_12345xyz")
	for _, itemID := range []*string{nil, schemas.Ptr(""), schemas.Ptr("ws_123"), schemas.Ptr("rs_123"), schemas.Ptr("msg_123")} {
		encoded := encodeFunctionCallToolUseID(callID, itemID)
		if encoded != callID {
			t.Fatalf("encode(%q, %v) = %v, want original callID pointer", *callID, itemID, encoded)
		}
	}
}

func TestFunctionCallToolUseIDEnvelopeDecodeFailOpen(t *testing.T) {
	t.Parallel()

	cases := []string{
		"call_plain",
		"call_abc" + functionCallIDEnvelopeMarker + "not-base64",
		"call_abc" + functionCallIDEnvelopeMarker + "e30", // {}
	}
	for _, raw := range cases {
		t.Run(raw, func(t *testing.T) {
			t.Parallel()
			callID, itemID, ok := decodeFunctionCallToolUseID(&raw)
			if ok {
				t.Fatal("decode unexpectedly succeeded")
			}
			if callID == nil || *callID != raw {
				t.Fatalf("callID = %v, want original %q", callID, raw)
			}
			if itemID != nil {
				t.Fatalf("itemID = %v, want nil", itemID)
			}
		})
	}
}

func TestToAnthropicResponsesResponseEncodesFunctionCallItemID(t *testing.T) {
	t.Parallel()

	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()

	resp := &schemas.BifrostResponsesResponse{
		ID:    schemas.Ptr("resp_original"),
		Model: "gpt-5.5",
		Output: []schemas.ResponsesMessage{{
			ID:   schemas.Ptr("fc_original"),
			Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall),
			ResponsesToolMessage: &schemas.ResponsesToolMessage{
				CallID:    schemas.Ptr("call_original"),
				Name:      schemas.Ptr("Bash"),
				Arguments: schemas.Ptr(`{"command":"pwd"}`),
			},
		}},
	}

	anthropicResp := ToAnthropicResponsesResponse(ctx, resp)
	if anthropicResp == nil || len(anthropicResp.Content) != 1 {
		t.Fatalf("response content len = %d, want 1", len(anthropicResp.Content))
	}
	block := anthropicResp.Content[0]
	if block.ID == nil || *block.ID == "call_original" {
		t.Fatalf("block.ID = %v, want encoded ID", block.ID)
	}
	callID, itemID, ok := decodeFunctionCallToolUseID(block.ID)
	if !ok {
		t.Fatalf("encoded ID did not decode: %q", *block.ID)
	}
	if callID == nil || *callID != "call_original" {
		t.Fatalf("decoded callID = %v, want call_original", callID)
	}
	if itemID == nil || *itemID != "fc_original" {
		t.Fatalf("decoded itemID = %v, want fc_original", itemID)
	}
}

func TestConvertBifrostFunctionCallToAnthropicToolUseNoItemIDKeepsCallID(t *testing.T) {
	t.Parallel()

	msg := &schemas.ResponsesMessage{
		Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall),
		ResponsesToolMessage: &schemas.ResponsesToolMessage{
			CallID:    schemas.Ptr("call_original"),
			Name:      schemas.Ptr("Bash"),
			Arguments: schemas.Ptr(`{"command":"pwd"}`),
		},
	}

	block := convertBifrostFunctionCallToAnthropicToolUse(nil, msg)
	if block == nil || block.ID == nil || *block.ID != "call_original" {
		t.Fatalf("block.ID = %v, want call_original", block)
	}
}

func TestAnthropicEncodedToolUseRestoresFunctionCallItemID(t *testing.T) {
	t.Parallel()

	encodedID := encodeFunctionCallToolUseID(schemas.Ptr("call_original"), schemas.Ptr("fc_original"))
	messages := []AnthropicMessage{{
		Role: AnthropicMessageRoleAssistant,
		Content: AnthropicContent{ContentBlocks: []AnthropicContentBlock{{
			Type:  AnthropicContentBlockTypeToolUse,
			ID:    encodedID,
			Name:  schemas.Ptr("Bash"),
			Input: []byte(`{"command":"pwd"}`),
		}}},
	}}

	converted := ConvertAnthropicMessagesToBifrostMessages(nil, messages, nil, true, false)
	if len(converted) != 1 {
		t.Fatalf("converted len = %d, want 1", len(converted))
	}
	msg := converted[0]
	if msg.ID == nil || *msg.ID != "fc_original" {
		t.Fatalf("function_call id = %v, want fc_original", msg.ID)
	}
	if msg.ResponsesToolMessage == nil || msg.ResponsesToolMessage.CallID == nil || *msg.ResponsesToolMessage.CallID != "call_original" {
		t.Fatalf("callID = %#v, want call_original", msg.ResponsesToolMessage)
	}
}

func TestAnthropicEncodedToolResultDropsFunctionCallItemID(t *testing.T) {
	t.Parallel()

	encodedID := encodeFunctionCallToolUseID(schemas.Ptr("call_original"), schemas.Ptr("fc_original"))
	messages := []AnthropicMessage{{
		Role: AnthropicMessageRoleUser,
		Content: AnthropicContent{ContentBlocks: []AnthropicContentBlock{{
			Type:      AnthropicContentBlockTypeToolResult,
			ToolUseID: encodedID,
			Content:   &AnthropicContent{ContentStr: schemas.Ptr("ok")},
		}}},
	}}

	converted := ConvertAnthropicMessagesToBifrostMessages(nil, messages, nil, true, false)
	if len(converted) != 1 {
		t.Fatalf("converted len = %d, want 1", len(converted))
	}
	msg := converted[0]
	if msg.ID != nil {
		t.Fatalf("function_call_output id = %v, want nil", msg.ID)
	}
	if msg.ResponsesToolMessage == nil || msg.ResponsesToolMessage.CallID == nil || *msg.ResponsesToolMessage.CallID != "call_original" {
		t.Fatalf("callID = %#v, want call_original", msg.ResponsesToolMessage)
	}
}

func TestGenericBifrostFunctionCallAndOutputKeepRawAnthropicID(t *testing.T) {
	t.Parallel()

	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()

	bifrostMessages := []schemas.ResponsesMessage{
		{
			ID:     schemas.Ptr("fc_original"),
			Type:   schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall),
			Status: schemas.Ptr("completed"),
			ResponsesToolMessage: &schemas.ResponsesToolMessage{
				CallID:    schemas.Ptr("call_original"),
				Name:      schemas.Ptr("Bash"),
				Arguments: schemas.Ptr(`{"command":"pwd"}`),
			},
		},
		{
			Type:   schemas.Ptr(schemas.ResponsesMessageTypeFunctionCallOutput),
			Status: schemas.Ptr("completed"),
			ResponsesToolMessage: &schemas.ResponsesToolMessage{
				CallID: schemas.Ptr("call_original"),
				Output: &schemas.ResponsesToolMessageOutputStruct{
					ResponsesToolCallOutputStr: schemas.Ptr("ok"),
				},
			},
		},
	}

	anthropicMessages, _ := ConvertBifrostMessagesToAnthropicMessages(ctx, bifrostMessages, true, schemas.Anthropic, "claude-opus-4-8")
	if len(anthropicMessages) != 2 {
		t.Fatalf("anthropic messages len = %d, want 2", len(anthropicMessages))
	}
	toolUse := anthropicMessages[0].Content.ContentBlocks[0]
	toolResult := anthropicMessages[1].Content.ContentBlocks[0]
	if toolUse.ID == nil || *toolUse.ID != "call_original" {
		t.Fatalf("tool_use.id = %v, want raw call_original", toolUse.ID)
	}
	if toolResult.ToolUseID == nil || *toolResult.ToolUseID != "call_original" {
		t.Fatalf("tool_result.tool_use_id = %v, want raw call_original", toolResult.ToolUseID)
	}
}

func TestNativeWebSearchCallDoesNotUseFunctionCallEnvelope(t *testing.T) {
	t.Parallel()

	query := "interesting facts about tardigrades"
	msg := &schemas.ResponsesMessage{
		ID:   schemas.Ptr("ws_original"),
		Type: schemas.Ptr(schemas.ResponsesMessageTypeWebSearchCall),
		ResponsesToolMessage: &schemas.ResponsesToolMessage{
			CallID: schemas.Ptr("ws_original"),
			Action: &schemas.ResponsesToolMessageActionStruct{
				ResponsesWebSearchToolCallAction: &schemas.ResponsesWebSearchToolCallAction{
					Type:  "search",
					Query: &query,
					Sources: []schemas.ResponsesWebSearchToolCallActionSearchSource{{
						Type: "url",
						URL:  "https://example.com",
					}},
				},
			},
		},
	}

	blocks := convertBifrostWebSearchCallToAnthropicBlocks(msg)
	if len(blocks) != 2 {
		t.Fatalf("blocks len = %d, want 2", len(blocks))
	}
	if blocks[0].ID == nil || *blocks[0].ID != "ws_original" || strings.Contains(*blocks[0].ID, functionCallIDEnvelopeMarker) {
		t.Fatalf("server_tool_use id = %v, want plain ws_original", blocks[0].ID)
	}
	if blocks[1].ToolUseID == nil || *blocks[1].ToolUseID != "ws_original" || strings.Contains(*blocks[1].ToolUseID, functionCallIDEnvelopeMarker) {
		t.Fatalf("web_search_tool_result tool_use_id = %v, want plain ws_original", blocks[1].ToolUseID)
	}
}
