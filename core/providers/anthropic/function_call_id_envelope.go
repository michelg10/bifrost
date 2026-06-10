package anthropic

import (
	"encoding/base64"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
)

const functionCallIDEnvelopeMarker = "__bfrfc1__"

type functionCallIDEnvelope struct {
	Version int    `json:"v"`
	CallID  string `json:"call_id"`
	ItemID  string `json:"item_id"`
}

func encodeFunctionCallToolUseID(callID *string, itemID *string) *string {
	if callID == nil || *callID == "" {
		return callID
	}
	if itemID == nil || *itemID == "" || !strings.HasPrefix(*itemID, "fc_") {
		return callID
	}
	if strings.Contains(*callID, functionCallIDEnvelopeMarker) {
		return callID
	}

	payload, err := sonic.Marshal(functionCallIDEnvelope{
		Version: 1,
		CallID:  *callID,
		ItemID:  *itemID,
	})
	if err != nil {
		return callID
	}

	encoded := *callID + functionCallIDEnvelopeMarker + base64.RawURLEncoding.EncodeToString(payload)
	return schemas.Ptr(encoded)
}

func decodeFunctionCallToolUseID(raw *string) (callID *string, itemID *string, decoded bool) {
	if raw == nil || *raw == "" {
		return raw, nil, false
	}

	prefix, encoded, ok := strings.Cut(*raw, functionCallIDEnvelopeMarker)
	if !ok || prefix == "" || encoded == "" {
		return raw, nil, false
	}

	payload, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		return raw, nil, false
	}

	var envelope functionCallIDEnvelope
	if err := sonic.Unmarshal(payload, &envelope); err != nil {
		return raw, nil, false
	}
	if envelope.Version != 1 || envelope.CallID == "" || envelope.ItemID == "" {
		return raw, nil, false
	}
	if envelope.CallID != prefix || !strings.HasPrefix(envelope.ItemID, "fc_") {
		return raw, nil, false
	}

	return schemas.Ptr(envelope.CallID), schemas.Ptr(envelope.ItemID), true
}
