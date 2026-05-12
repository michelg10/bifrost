package anthropic

import (
	"encoding/base64"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
)

const bifrostReasoningEnvelopePrefix = "bfrs1."
const openAIReasoningEncryptedContentInclude = "reasoning.encrypted_content"

type bifrostReasoningEnvelope struct {
	Version          int                                 `json:"v"`
	Provider         schemas.ModelProvider               `json:"provider,omitempty"`
	ResponseID       *string                             `json:"response_id,omitempty"`
	ReasoningID      *string                             `json:"reasoning_id,omitempty"`
	Status           *string                             `json:"status,omitempty"`
	Summary          []schemas.ResponsesReasoningSummary `json:"summary,omitempty"`
	EncryptedContent *string                             `json:"encrypted_content,omitempty"`
}

func encodeBifrostReasoningEnvelope(envelope bifrostReasoningEnvelope) (string, error) {
	envelope.Version = 1
	data, err := sonic.Marshal(envelope)
	if err != nil {
		return "", err
	}
	return bifrostReasoningEnvelopePrefix + base64.RawURLEncoding.EncodeToString(data), nil
}

func decodeBifrostReasoningEnvelope(value *string) (*bifrostReasoningEnvelope, bool) {
	if value == nil || !strings.HasPrefix(*value, bifrostReasoningEnvelopePrefix) {
		return nil, false
	}
	data, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(*value, bifrostReasoningEnvelopePrefix))
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

func supportsOpenAIReasoningEnvelope(provider schemas.ModelProvider) bool {
	return provider == schemas.OpenAI || provider == schemas.Azure
}
