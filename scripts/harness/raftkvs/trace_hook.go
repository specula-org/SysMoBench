// Trace hook injected into data/repositories/pgo/systems/raftkvs/bootstrap/
// by scripts/harness/raftkvs/run.sh. Holds a package-global trace.Recorder
// that bootstrap.newServerCtxs + bootstrap.NewClient reference; a nil value
// (the default) is a no-op thanks to distsys.CommitEvent's nil check.

package bootstrap

import "github.com/DistCompiler/pgo/distsys/trace"

// TraceRecorder, if non-nil, receives every MPCal block execution from
// every server / client archetype in this process. Typically set once
// from a test function before bootstrap.NewServer / bootstrap.NewClient.
var TraceRecorder trace.Recorder
