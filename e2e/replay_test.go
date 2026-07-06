//go:build e2e
// +build e2e

package e2e

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"
)

// worldIDPattern matches the `WORLD_ID="<id>"` line the recorded scripts
// use to select which world `ww new` creates against. Harness parses this
// to know which fixture world to seed on the target server.
var worldIDPattern = regexp.MustCompile(`^WORLD_ID="([^"]+)"`)

// gameURLPattern matches the "Game viewable at: <url>" line the scripts
// echo after `ww new`. Harness parses it in watch mode to auto-open the
// URL in a browser tab so the user can watch the replay drive the game.
var gameURLPattern = regexp.MustCompile(`^Game viewable at: (\S+)`)

// TestReplayScripts runs every replays/*.sh script against the server
// pointed at by LILBATTLE_E2E_SERVER. Each subtest ensures its fixture
// world exists on the target, then executes the script. The scripts' own
// `set -e` and `trap 'echo FAILED at line …' ERR` bubble the specific
// failing ww command into the test output.
//
// Watch mode: LILBATTLE_E2E_WATCH=true prints the game URL and (on
// macOS / Linux) invokes `open` / `xdg-open` so the user watches the
// replay live in a browser tab. Off by default.
func TestReplayScripts(t *testing.T) {
	server := serverURL(t)
	wwDir := wwPathDir(t)

	scriptsDir := filepath.Join(moduleRoot(t), "replays")
	entries, err := os.ReadDir(scriptsDir)
	if err != nil {
		t.Fatalf("read scripts dir: %v", err)
	}

	var scripts []string
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".sh") {
			scripts = append(scripts, entry.Name())
		}
	}
	if len(scripts) == 0 {
		t.Fatal("no *.sh replay scripts found under replays/")
	}

	for _, scriptName := range scripts {
		scriptName := scriptName // capture for parallel
		t.Run(strings.TrimSuffix(scriptName, ".sh"), func(t *testing.T) {
			scriptPath := filepath.Join(scriptsDir, scriptName)
			worldID, err := extractWorldID(scriptPath)
			if err != nil {
				t.Fatalf("extract WORLD_ID from %s: %v", scriptName, err)
			}

			_ = worldID // parsed only to fail early if a script is malformed
			runReplayScript(t, scriptPath, server, wwDir)
		})
	}
}

// extractWorldID scans a replay script for its WORLD_ID declaration.
// Every recorded script embeds one on a line like `WORLD_ID="7e5016a4"`
// near the top; the harness reads it once to know which fixture to seed.
func extractWorldID(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		if m := worldIDPattern.FindStringSubmatch(scanner.Text()); m != nil {
			return m[1], nil
		}
	}
	if err := scanner.Err(); err != nil {
		return "", err
	}
	return "", fmt.Errorf("no WORLD_ID declaration found")
}

// runReplayScript executes one .sh under bash, tailing stdout for the
// game-URL line (used by watch mode). On failure, the captured output
// is dumped so the exact `ww` command and its error message reach the
// test log — matching the scripts' own `trap 'echo FAILED at line …'`.
func runReplayScript(t *testing.T, scriptPath, serverBase, wwDir string) {
	t.Helper()

	// exec.Command("bash", scriptPath) runs the script as bash's argv[0],
	// NOT as `bash -c "<string>"`. There's no shell-string interpolation
	// of an outside value here; bash reads the file directly. The path
	// itself is bounded to replays/*.sh via os.ReadDir + suffix filter,
	// not caller input.
	cmd := exec.Command("bash", scriptPath)
	cmd.Env = append(os.Environ(),
		"LILBATTLE_SERVER="+serverBase,
		"PATH="+wwDir+string(os.PathListSeparator)+os.Getenv("PATH"),
		"LILBATTLE_CONFIRM=false",
	)

	var stdoutBuf bytes.Buffer
	stdoutR, stdoutW := io.Pipe()
	cmd.Stdout = stdoutW
	cmd.Stderr = &stdoutBuf // interleave stderr into the same buffer for context

	if err := cmd.Start(); err != nil {
		t.Fatalf("start bash: %v", err)
	}

	// Consume the script's stdout, echo to the test-scoped buffer, and
	// react to the game-URL line in watch mode. Running in a goroutine so
	// bash never blocks on a full pipe.
	watchEnabled := os.Getenv("LILBATTLE_E2E_WATCH") == "true"
	go func() {
		defer stdoutR.Close()
		scanner := bufio.NewScanner(stdoutR)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			stdoutBuf.WriteString(line + "\n")
			if watchEnabled {
				if m := gameURLPattern.FindStringSubmatch(line); m != nil {
					openInBrowser(t, m[1])
				}
			}
		}
	}()

	err := cmd.Wait()
	_ = stdoutW.Close()
	time.Sleep(50 * time.Millisecond)

	if err != nil {
		t.Logf("--- script output ---\n%s", stdoutBuf.String())
		t.Fatalf("replay %s failed: %v", filepath.Base(scriptPath), err)
	}
}

// openInBrowser invokes the OS's browser-launch command on the URL. Only
// called in watch mode. Failures are logged but don't fail the test —
// watching the game is a debugging affordance, not a correctness signal.
func openInBrowser(t *testing.T, url string) {
	t.Helper()
	t.Logf("watch mode: opening %s", url)
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "linux":
		cmd = exec.Command("xdg-open", url)
	default:
		t.Logf("watch mode: no browser-launch command for %s; open manually: %s", runtime.GOOS, url)
		return
	}
	if err := cmd.Start(); err != nil {
		t.Logf("watch mode: browser launch failed: %v", err)
	}
}
