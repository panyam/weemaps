// Package e2e holds the recorded-replay integration harness. All real
// content compiles only under the `e2e` build tag; this file exists so
// default `go test ./tests/e2e/` succeeds with "no tests to run" instead
// of failing with "build constraints exclude all Go files in ...".
package e2e
