package anthropic

import (
	"encoding/base64"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
)

// anthropicClientRequestedModelKey stashes the model string the client (e.g.
// Claude Code) sent on the inbound /v1/messages request, captured before
// Bifrost resolves it to a provider deployment (alias -> e.g. azure/gpt-5.5).
// We echo this value back as the response `model` AND embed it in the
// reasoning-envelope signature so Anthropic clients see a self-consistent model
// identity across the conversation. Thinking signatures are model-bound: a
// block stamped with a foreign model than the session model is dropped on
// replay, breaking reasoning continuity.
const anthropicClientRequestedModelKey schemas.BifrostContextKey = "bifrost-anthropic-client-requested-model"

func setClientRequestedModel(ctx *schemas.BifrostContext, model string) {
	if ctx == nil || model == "" {
		return
	}
	// Set-once: the earliest capture (the raw client-facing model, taken before a
	// routing rule rewrites it to a backend deployment) must win over later
	// captures of the already-resolved model.
	if existing, ok := ctx.Value(anthropicClientRequestedModelKey).(string); ok && existing != "" {
		return
	}
	ctx.SetValue(anthropicClientRequestedModelKey, model)
}

func getClientRequestedModel(ctx *schemas.BifrostContext) string {
	if ctx == nil {
		return ""
	}
	// Prefer the original client-facing model captured by the governance HTTP
	// pre-hook before any routing rule rewrote it to a backend deployment.
	if v, ok := ctx.Value(schemas.BifrostContextKeyClientRequestedModel).(string); ok && v != "" {
		return v
	}
	// Fallback: the model captured at request conversion time. Correct when no
	// routing rule rewrote the model (req.Model is then the raw client model).
	if v, ok := ctx.Value(anthropicClientRequestedModelKey).(string); ok && v != "" {
		return v
	}
	return ""
}

type bifrostReasoningEnvelope struct {
	Version          int                                 `json:"v"`
	Model            *string                             `json:"model,omitempty"`
	Provider         schemas.ModelProvider               `json:"provider,omitempty"`
	ResponseID       *string                             `json:"response_id,omitempty"`
	ReasoningID      *string                             `json:"reasoning_id,omitempty"`
	Status           *string                             `json:"status,omitempty"`
	Summary          []schemas.ResponsesReasoningSummary `json:"summary,omitempty"`
	EncryptedContent *string                             `json:"encrypted_content,omitempty"`
}

// encodeBifrostReasoningEnvelope serializes the envelope into a signature that
// mimics a native Anthropic thinking signature: a prefix-less, standard-base64
// string whose decoded bytes embed the model id (via the Model field). The
// older "bfrs1." prefix is intentionally dropped so the signature is
// indistinguishable in shape from a real Claude signature, which itself encodes
// the model in its decoded payload.
func encodeBifrostReasoningEnvelope(envelope bifrostReasoningEnvelope) (string, error) {
	envelope.Version = 1
	data, err := sonic.Marshal(envelope)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(data), nil
}

// decodeBifrostReasoningEnvelope recognizes our signatures structurally rather
// than by prefix: standard-base64 decode, then JSON-unmarshal, then require our
// version + reasoning id. A real Anthropic native signature base64-decodes to
// protobuf (not JSON) and fails the unmarshal, so it's correctly treated as
// native and passed through.
func decodeBifrostReasoningEnvelope(value *string) (*bifrostReasoningEnvelope, bool) {
	if value == nil || *value == "" {
		return nil, false
	}
	data, err := base64.StdEncoding.DecodeString(*value)
	if err != nil {
		return nil, false
	}
	var envelope bifrostReasoningEnvelope
	if err := sonic.Unmarshal(data, &envelope); err != nil {
		return nil, false
	}
	if envelope.Version != 1 || envelope.ReasoningID == nil || *envelope.ReasoningID == "" {
		return nil, false
	}
	return &envelope, true
}

func appendUniqueInclude(params *schemas.ResponsesParameters, include string) {
	if params == nil || include == "" {
		return
	}
	for _, existing := range params.Include {
		if existing == include {
			return
		}
	}
	params.Include = append(params.Include, include)
}

// responseOutputHasToolUse reports whether a completed Responses response carries
// any tool/function call in its output. Used to derive the Anthropic stop reason
// (`tool_use` vs `end_turn`) on the streaming path, where the OpenAI Responses API
// provides no top-level stop reason.
func responseOutputHasToolUse(resp *schemas.BifrostResponsesResponse) bool {
	if resp == nil {
		return false
	}
	for i := range resp.Output {
		t := resp.Output[i].Type
		if t == nil {
			continue
		}
		switch *t {
		case schemas.ResponsesMessageTypeFunctionCall,
			schemas.ResponsesMessageTypeCustomToolCall,
			schemas.ResponsesMessageTypeLocalShellCall,
			schemas.ResponsesMessageTypeComputerCall,
			schemas.ResponsesMessageTypeMCPCall:
			return true
		}
	}
	return false
}

func supportsOpenAIReasoningEnvelope(provider schemas.ModelProvider) bool {
	return provider == schemas.OpenAI || provider == schemas.Azure
}
