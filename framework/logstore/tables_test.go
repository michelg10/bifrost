package logstore

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDeserializeFieldsReconstructsTokenUsageFromDenormalizedColumns(t *testing.T) {
	log := &Log{
		PromptTokens:     10,
		CompletionTokens: 5,
		TotalTokens:      15,
	}
	require.NoError(t, log.DeserializeFields())
	require.NotNil(t, log.TokenUsageParsed)
	assert.Equal(t, 10, log.TokenUsageParsed.PromptTokens)
	assert.Equal(t, 5, log.TokenUsageParsed.CompletionTokens)
	assert.Equal(t, 15, log.TokenUsageParsed.TotalTokens)
	assert.Nil(t, log.TokenUsageParsed.PromptTokensDetails)
}

func TestDeserializeFieldsPrefersSerializedTokenUsage(t *testing.T) {
	log := &Log{
		TokenUsage:       `{"prompt_tokens":99,"completion_tokens":1,"total_tokens":100}`,
		PromptTokens:     10,
		CompletionTokens: 5,
		TotalTokens:      15,
	}
	require.NoError(t, log.DeserializeFields())
	require.NotNil(t, log.TokenUsageParsed)
	assert.Equal(t, 99, log.TokenUsageParsed.PromptTokens)
	assert.Equal(t, 1, log.TokenUsageParsed.CompletionTokens)
	assert.Equal(t, 100, log.TokenUsageParsed.TotalTokens)
}

func TestDeserializeFieldsDoesNotReconstructTokenUsageWhenSerializedValueIsMalformed(t *testing.T) {
	log := &Log{
		TokenUsage:       `{"prompt_tokens":`,
		PromptTokens:     10,
		CompletionTokens: 5,
		TotalTokens:      15,
	}
	require.NoError(t, log.DeserializeFields())
	assert.Nil(t, log.TokenUsageParsed)
}

func TestDeserializeFieldsSkipsTokenUsageReconstructionWhenAllZero(t *testing.T) {
	log := &Log{}
	require.NoError(t, log.DeserializeFields())
	assert.Nil(t, log.TokenUsageParsed)
}

// TestSanitizeJSONForJSONB locks in the LOCAL FORK PATCH to
// sanitizeJSONForJSONB. The pre-patch implementation used naive
// strings.ReplaceAll and produced corrupt JSON for any payload that
// contained a properly-escaped backslash followed by "u0000".
//
// If/when upstream Bifrost ships their own fix and we drop the local
// patch on the next merge, this test should still pass against the
// upstream sanitizer. If it does not, do NOT revert the local patch.
func TestSanitizeJSONForJSONB(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{
			name: `no escape sequences pass through unchanged`,
			in:   `{"text":"hello world"}`,
			want: `{"text":"hello world"}`,
		},
		{
			name: `lone null escape is stripped`,
			in:   `{"text":"a\u0000b"}`,
			want: `{"text":"ab"}`,
		},
		{
			name: `lone uppercase null escape is stripped`,
			in:   `{"text":"a\U0000b"}`,
			want: `{"text":"ab"}`,
		},
		{
			name: `escaped backslash followed by u0000 is preserved (regression)`,
			in:   `{"text":"a\\u0000b"}`,
			want: `{"text":"a\\u0000b"}`,
		},
		{
			name: `escaped backslash followed by U0000 is preserved (regression)`,
			in:   `{"text":"a\\U0000b"}`,
			want: `{"text":"a\\U0000b"}`,
		},
		{
			name: `mixed: real null escape AND escaped-backslash + u0000 in one string`,
			in:   `{"text":"x\u0000y\\u0000z\u0000w"}`,
			want: `{"text":"xy\\u0000zw"}`,
		},
		{
			name: `four backslashes followed by u0000 are all preserved (regression)`,
			in:   `{"text":"\\\\u0000"}`,
			want: `{"text":"\\\\u0000"}`,
		},
		{
			name: `three backslashes + u0000 strips only the trailing null escape`,
			in:   `{"text":"\\\u0000"}`,
			want: `{"text":"\\"}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := sanitizeJSONForJSONB(tc.in)
			if got != tc.want {
				t.Errorf("sanitizeJSONForJSONB(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
