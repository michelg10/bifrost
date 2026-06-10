package anthropic

import (
	"context"
	"encoding/base64"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

func TestBifrostReasoningEnvelopeRoundTrip(t *testing.T) {
	reasoningID := "rs_123"
	status := "completed"
	responseID := "resp_123"
	encrypted := "enc_123"
	model := "claude-opus-4-8"
	envelope := bifrostReasoningEnvelope{
		Model:            &model,
		ResponseID:       &responseID,
		ReasoningID:      &reasoningID,
		Status:           &status,
		EncryptedContent: &encrypted,
		Summary: []schemas.ResponsesReasoningSummary{{
			Type: schemas.ResponsesReasoningContentBlockTypeSummaryText,
			Text: "summary",
		}},
	}

	encoded, err := encodeBifrostReasoningEnvelope(envelope)
	if err != nil {
		t.Fatalf("encodeBifrostReasoningEnvelope returned error: %v", err)
	}
	// The signature mimics a native Anthropic signature: prefix-less base64 whose
	// decoded bytes embed the model. It must NOT carry the legacy "bfrs1." tell.
	if strings.HasPrefix(encoded, "bfrs1.") {
		t.Fatalf("encoded envelope should not carry the bfrs1. prefix: %q", encoded)
	}
	if raw, derr := base64.StdEncoding.DecodeString(encoded); derr != nil {
		t.Fatalf("signature is not standard base64: %v", derr)
	} else if !strings.Contains(string(raw), model) {
		t.Fatalf("decoded signature does not embed the model %q: %s", model, raw)
	}

	decoded, ok := decodeBifrostReasoningEnvelope(&encoded)
	if !ok {
		t.Fatal("expected envelope to decode")
	}
	if decoded.Model == nil || *decoded.Model != model {
		t.Fatalf("model = %v, want %q", decoded.Model, model)
	}
	if decoded.ReasoningID == nil || *decoded.ReasoningID != reasoningID {
		t.Fatalf("reasoning id = %v, want %q", decoded.ReasoningID, reasoningID)
	}
	if decoded.Status == nil || *decoded.Status != status {
		t.Fatalf("status = %v, want %q", decoded.Status, status)
	}
	if decoded.ResponseID == nil || *decoded.ResponseID != responseID {
		t.Fatalf("response id = %v, want %q", decoded.ResponseID, responseID)
	}
	if decoded.EncryptedContent == nil || *decoded.EncryptedContent != encrypted {
		t.Fatalf("encrypted content = %v, want %q", decoded.EncryptedContent, encrypted)
	}
	if len(decoded.Summary) != 1 || decoded.Summary[0].Text != "summary" {
		t.Fatalf("summary = %#v, want one summary", decoded.Summary)
	}
}

func TestDecodeBifrostReasoningEnvelopeIgnoresNativeSignature(t *testing.T) {
	native := "anthropic_native_signature"
	if _, ok := decodeBifrostReasoningEnvelope(&native); ok {
		t.Fatal("expected native Anthropic signature not to decode")
	}
}

func TestAnthropicThinkingEnvelopeRestoresReasoningMessage(t *testing.T) {
	reasoningID := "rs_restore"
	status := "completed"
	encrypted := "enc_restore"
	summary := []schemas.ResponsesReasoningSummary{{
		Type: schemas.ResponsesReasoningContentBlockTypeSummaryText,
		Text: "visible summary",
	}}
	encoded, err := encodeBifrostReasoningEnvelope(bifrostReasoningEnvelope{
		ReasoningID:      &reasoningID,
		Status:           &status,
		Summary:          summary,
		EncryptedContent: &encrypted,
	})
	if err != nil {
		t.Fatalf("encodeBifrostReasoningEnvelope returned error: %v", err)
	}

	block := AnthropicContentBlock{
		Type:      AnthropicContentBlockTypeThinking,
		Thinking:  schemas.Ptr("visible summary"),
		Signature: &encoded,
	}
	msg := convertAnthropicThinkingEnvelopeToBifrostReasoning(&block)
	if msg == nil {
		t.Fatal("expected reasoning message")
	}
	if msg.ID == nil || *msg.ID != reasoningID {
		t.Fatalf("id = %v, want %q", msg.ID, reasoningID)
	}
	if msg.Status == nil || *msg.Status != status {
		t.Fatalf("status = %v, want %q", msg.Status, status)
	}
	if msg.ResponsesReasoning == nil || msg.ResponsesReasoning.EncryptedContent == nil || *msg.ResponsesReasoning.EncryptedContent != encrypted {
		t.Fatalf("encrypted content not restored: %#v", msg.ResponsesReasoning)
	}
	if len(msg.ResponsesReasoning.Summary) != 1 || msg.ResponsesReasoning.Summary[0].Text != "visible summary" {
		t.Fatalf("summary = %#v", msg.ResponsesReasoning.Summary)
	}
}

func TestToBifrostResponsesRequestReasoningAddsEncryptedContentIncludeAndStore(t *testing.T) {
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()

	req := &AnthropicMessageRequest{
		Model:     "azure/gpt-5.5-cc",
		MaxTokens: 1024,
		Thinking:  &AnthropicThinking{Type: "enabled"},
		Messages: []AnthropicMessage{{
			Role: AnthropicMessageRoleUser,
			Content: AnthropicContent{
				ContentStr: schemas.Ptr("hello"),
			},
		}},
	}

	got := req.ToBifrostResponsesRequest(ctx)
	if got == nil || got.Params == nil {
		t.Fatal("expected params")
	}
	found := false
	for _, include := range got.Params.Include {
		if include == "reasoning.encrypted_content" {
			found = true
		}
	}
	if !found {
		t.Fatalf("include = %#v, want reasoning.encrypted_content", got.Params.Include)
	}
	if got.Params.Store == nil || !*got.Params.Store {
		t.Fatalf("store = %#v, want explicit true", got.Params.Store)
	}
}

func TestToBifrostResponsesRequestNoReasoningStillAddsIncludeAndStore(t *testing.T) {
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()

	// Some OpenAI models are reasoning-only ("none" is rejected, e.g. -codex),
	// so the include + store belt-and-suspenders applies even when thinking is
	// off; ToOpenAIResponsesRequest strips the include for models that cannot
	// return encrypted content.
	for name, thinking := range map[string]*AnthropicThinking{
		"thinking absent":   nil,
		"thinking disabled": {Type: "disabled"},
	} {
		req := &AnthropicMessageRequest{
			Model:     "azure/gpt-5.5-cc",
			MaxTokens: 1024,
			Thinking:  thinking,
			Messages: []AnthropicMessage{{
				Role: AnthropicMessageRoleUser,
				Content: AnthropicContent{
					ContentStr: schemas.Ptr("hello"),
				},
			}},
		}

		got := req.ToBifrostResponsesRequest(ctx)
		if got == nil || got.Params == nil {
			t.Fatalf("%s: expected params", name)
		}
		found := false
		for _, include := range got.Params.Include {
			if include == "reasoning.encrypted_content" {
				found = true
			}
		}
		if !found {
			t.Fatalf("%s: include = %#v, want reasoning.encrypted_content", name, got.Params.Include)
		}
		if got.Params.Store == nil || !*got.Params.Store {
			t.Fatalf("%s: store = %#v, want explicit true", name, got.Params.Store)
		}
	}
}

func TestToBifrostResponsesRequestCountTokensSkipsEncryptedContentIncludeAndStore(t *testing.T) {
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()
	ctx.SetValue(schemas.BifrostContextKeyHTTPRequestType, schemas.CountTokensRequest)

	req := &AnthropicMessageRequest{
		Model:     "azure/gpt-5.5-cc",
		MaxTokens: 1024,
		Thinking:  &AnthropicThinking{Type: "enabled"},
		Messages: []AnthropicMessage{{
			Role: AnthropicMessageRoleUser,
			Content: AnthropicContent{
				ContentStr: schemas.Ptr("hello"),
			},
		}},
	}

	got := req.ToBifrostResponsesRequest(ctx)
	if got == nil || got.Params == nil {
		t.Fatal("expected params")
	}
	if len(got.Params.Include) != 0 {
		t.Fatalf("include = %#v, want empty for count_tokens", got.Params.Include)
	}
	if got.Params.Store != nil {
		t.Fatalf("store = %#v, want unset for count_tokens", got.Params.Store)
	}
}

func TestToBifrostResponsesRequestClientIncludeNotDuplicated(t *testing.T) {
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()

	req := &AnthropicMessageRequest{
		Model:     "azure/gpt-5.5-cc",
		MaxTokens: 1024,
		Thinking:  &AnthropicThinking{Type: "enabled"},
		ExtraParams: map[string]interface{}{
			"include": []string{"reasoning.encrypted_content"},
		},
		Messages: []AnthropicMessage{{
			Role: AnthropicMessageRoleUser,
			Content: AnthropicContent{
				ContentStr: schemas.Ptr("hello"),
			},
		}},
	}

	got := req.ToBifrostResponsesRequest(ctx)
	if got == nil || got.Params == nil {
		t.Fatal("expected params")
	}
	count := 0
	for _, include := range got.Params.Include {
		if include == "reasoning.encrypted_content" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("include = %#v, want exactly one reasoning.encrypted_content", got.Params.Include)
	}
}

func TestToBifrostResponsesRequestClientStoreOverrideWins(t *testing.T) {
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()

	req := &AnthropicMessageRequest{
		Model:     "azure/gpt-5.5-cc",
		MaxTokens: 1024,
		Thinking:  &AnthropicThinking{Type: "enabled"},
		ExtraParams: map[string]interface{}{
			"store": false,
		},
		Messages: []AnthropicMessage{{
			Role: AnthropicMessageRoleUser,
			Content: AnthropicContent{
				ContentStr: schemas.Ptr("hello"),
			},
		}},
	}

	got := req.ToBifrostResponsesRequest(ctx)
	if got == nil || got.Params == nil {
		t.Fatal("expected params")
	}
	if got.Params.Store == nil || *got.Params.Store {
		t.Fatalf("store = %#v, want explicit false from client override", got.Params.Store)
	}
}

func TestToAnthropicResponsesResponseEmitsReasoningEnvelope(t *testing.T) {
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()

	reasoningID := "rs_nonstream"
	status := "completed"
	encrypted := "enc_nonstream"
	respID := "resp_nonstream"
	resp := &schemas.BifrostResponsesResponse{
		ID:    &respID,
		Model: "gpt-5.5-cc",
		ExtraFields: schemas.BifrostResponseExtraFields{
			Provider: schemas.Azure,
		},
		Output: []schemas.ResponsesMessage{{
			ID:     &reasoningID,
			Type:   schemas.Ptr(schemas.ResponsesMessageTypeReasoning),
			Status: &status,
			ResponsesReasoning: &schemas.ResponsesReasoning{
				Summary: []schemas.ResponsesReasoningSummary{{
					Type: schemas.ResponsesReasoningContentBlockTypeSummaryText,
					Text: "summary",
				}},
				EncryptedContent: &encrypted,
			},
		}},
	}

	got := ToAnthropicResponsesResponse(ctx, resp)
	if len(got.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(got.Content))
	}
	block := got.Content[0]
	if block.Type != AnthropicContentBlockTypeThinking || block.Signature == nil {
		t.Fatalf("block = %#v, want thinking with signature", block)
	}
	decoded, ok := decodeBifrostReasoningEnvelope(block.Signature)
	if !ok {
		t.Fatal("expected signature envelope")
	}
	if decoded.ReasoningID == nil || *decoded.ReasoningID != reasoningID {
		t.Fatalf("reasoning id = %v, want %q", decoded.ReasoningID, reasoningID)
	}
}

func TestToAnthropicResponsesResponseEmptySummaryUsesThinkingSignatureEnvelope(t *testing.T) {
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()

	reasoningID := "rs_empty_summary"
	encrypted := "enc_empty_summary"
	resp := &schemas.BifrostResponsesResponse{
		Model: "gpt-5.5-cc",
		ExtraFields: schemas.BifrostResponseExtraFields{
			Provider: schemas.Azure,
		},
		Output: []schemas.ResponsesMessage{{
			ID:   &reasoningID,
			Type: schemas.Ptr(schemas.ResponsesMessageTypeReasoning),
			ResponsesReasoning: &schemas.ResponsesReasoning{
				Summary:          []schemas.ResponsesReasoningSummary{},
				EncryptedContent: &encrypted,
			},
		}},
	}

	got := ToAnthropicResponsesResponse(ctx, resp)
	if len(got.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(got.Content))
	}
	block := got.Content[0]
	if block.Type != AnthropicContentBlockTypeThinking {
		t.Fatalf("block type = %q, want thinking", block.Type)
	}
	if block.Thinking == nil || *block.Thinking != "" {
		t.Fatalf("thinking = %v, want empty string", block.Thinking)
	}
	decoded, ok := decodeBifrostReasoningEnvelope(block.Signature)
	if !ok {
		t.Fatal("expected signature envelope")
	}
	if decoded.ReasoningID == nil || *decoded.ReasoningID != reasoningID {
		t.Fatalf("reasoning id = %v, want %q", decoded.ReasoningID, reasoningID)
	}
	if decoded.EncryptedContent == nil || *decoded.EncryptedContent != encrypted {
		t.Fatalf("encrypted content = %v, want %q", decoded.EncryptedContent, encrypted)
	}
}

func TestToAnthropicResponsesStreamResponseEmitsReasoningSignatureEnvelope(t *testing.T) {
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()

	reasoningID := "rs_stream"
	outputIndex := 0
	status := "completed"
	encrypted := "enc_stream"
	added := &schemas.BifrostResponsesStreamResponse{
		Type:        schemas.ResponsesStreamResponseTypeOutputItemAdded,
		OutputIndex: &outputIndex,
		ExtraFields: schemas.BifrostResponseExtraFields{
			Provider: schemas.Azure,
		},
		Item: &schemas.ResponsesMessage{
			ID:   &reasoningID,
			Type: schemas.Ptr(schemas.ResponsesMessageTypeReasoning),
			ResponsesReasoning: &schemas.ResponsesReasoning{
				Summary:          []schemas.ResponsesReasoningSummary{},
				EncryptedContent: &encrypted,
			},
		},
	}
	events := ToAnthropicResponsesStreamResponse(ctx, added)
	if len(events) != 1 || events[0].Type != AnthropicStreamEventTypeContentBlockStart {
		t.Fatalf("added events = %#v", events)
	}
	if events[0].ContentBlock == nil || events[0].ContentBlock.Type != AnthropicContentBlockTypeThinking {
		t.Fatalf("added content block = %#v, want thinking", events[0].ContentBlock)
	}

	delta := "stream summary"
	itemID := reasoningID
	deltaResp := &schemas.BifrostResponsesStreamResponse{
		Type:        schemas.ResponsesStreamResponseTypeReasoningSummaryTextDelta,
		OutputIndex: &outputIndex,
		ItemID:      &itemID,
		Delta:       &delta,
		ExtraFields: schemas.BifrostResponseExtraFields{
			Provider: schemas.Azure,
		},
	}
	if events := ToAnthropicResponsesStreamResponse(ctx, deltaResp); len(events) != 1 || events[0].Delta == nil || events[0].Delta.Thinking == nil {
		t.Fatalf("delta events = %#v", events)
	}

	done := &schemas.BifrostResponsesStreamResponse{
		Type:        schemas.ResponsesStreamResponseTypeOutputItemDone,
		OutputIndex: &outputIndex,
		ExtraFields: schemas.BifrostResponseExtraFields{
			Provider: schemas.Azure,
		},
		Item: &schemas.ResponsesMessage{
			ID:     &reasoningID,
			Type:   schemas.Ptr(schemas.ResponsesMessageTypeReasoning),
			Status: &status,
		},
	}
	doneEvents := ToAnthropicResponsesStreamResponse(ctx, done)
	if len(doneEvents) != 2 {
		t.Fatalf("done events len = %d, want 2: %#v", len(doneEvents), doneEvents)
	}
	if doneEvents[0].Type != AnthropicStreamEventTypeContentBlockDelta || doneEvents[0].Delta == nil || doneEvents[0].Delta.Signature == nil {
		t.Fatalf("first done event = %#v, want signature delta", doneEvents[0])
	}
	if doneEvents[1].Type != AnthropicStreamEventTypeContentBlockStop {
		t.Fatalf("second done event = %#v, want stop", doneEvents[1])
	}
	decoded, ok := decodeBifrostReasoningEnvelope(doneEvents[0].Delta.Signature)
	if !ok {
		t.Fatal("expected signature envelope")
	}
	if decoded.ReasoningID == nil || *decoded.ReasoningID != reasoningID {
		t.Fatalf("reasoning id = %v, want %q", decoded.ReasoningID, reasoningID)
	}
	if len(decoded.Summary) != 1 || decoded.Summary[0].Text != delta {
		t.Fatalf("summary = %#v, want %q", decoded.Summary, delta)
	}
}
