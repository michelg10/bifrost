package logstore

import "testing"

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
