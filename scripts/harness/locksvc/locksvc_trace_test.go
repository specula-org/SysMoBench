// TRACE-ENABLED variant of testNClients, 3 clients.
// Copied into data/repositories/pgo/systems/locksvc/ by run.sh.
// Writes PGo-native NDJSON traces to the file named by LOCKSVC_TRACE_FILE env var.

package locksvc_test

import (
	"fmt"
	"log"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/DistCompiler/pgo/distsys"
	"github.com/DistCompiler/pgo/distsys/resources"
	"github.com/DistCompiler/pgo/distsys/tla"
	"github.com/DistCompiler/pgo/distsys/trace"
	"github.com/DistCompiler/pgo/systems/locksvc"
)

// addr helper, matcher resource, runOrPanic, whileHoldingLock are reused
// from locksvc_test.go (same package).

func Test3ClientsWithTrace(t *testing.T) {
	tracePath := os.Getenv("LOCKSVC_TRACE_FILE")
	if tracePath == "" {
		t.Skip("LOCKSVC_TRACE_FILE not set")
	}
	_ = os.Remove(tracePath)
	f, err := os.Create(tracePath)
	if err != nil {
		t.Fatalf("cannot create trace file: %v", err)
	}
	defer f.Close()

	// Single recorder, shared across all archetypes. localFileRecorder is
	// mutex-serialized so interleaved writes are safe.
	recorder := trace.MakeLocalFileRecorder(f)

	log.Printf("waiting 3 seconds for prior tests to settle...")
	time.Sleep(3 * time.Second)

	const clientCount = 3

	srvId := tla.MakeNumber(0)
	srvCtx := distsys.NewMPCalContext(srvId, locksvc.AServer,
		distsys.EnsureArchetypeRefParam("network", resources.NewRelaxedMailboxes(addressFn(srvId))),
		distsys.SetTraceRecorder(recorder),
	)
	defer srvCtx.Stop()
	go runOrPanic(srvCtx)

	completionCh := make(chan struct{})
	var counter int
	var counterLock sync.Mutex

	for i := 0; i < clientCount; i++ {
		clientId := tla.MakeNumber(int32(i + 1))
		go func(cid tla.Value) {
			whileHoldingLockWithTrace(cid, recorder, func() {
				counterLock.Lock()
				counter++
				counterLock.Unlock()
				completionCh <- struct{}{}
			})
		}(clientId)
	}

	for i := 0; i < clientCount; i++ {
		<-completionCh
	}
	close(completionCh)

	if counter != clientCount {
		t.Errorf("expected %d acquires, got %d", clientCount, counter)
	} else {
		log.Printf("all %d clients acquired+released the lock", counter)
	}
}

// Shadow of whileHoldingLock that wires the trace recorder into the
// client context.
func whileHoldingLockWithTrace(clientId tla.Value, recorder trace.Recorder, body func()) {
	matcher := &matcherResource{}

	ctx := distsys.NewMPCalContext(clientId, locksvc.AClient,
		distsys.EnsureArchetypeRefParam("network", resources.NewRelaxedMailboxes(addressFn(clientId))),
		distsys.EnsureArchetypeRefParam("hasLock", resources.NewIncMap(func(index tla.Value) distsys.ArchetypeResource {
			if !index.Equal(clientId) {
				panic(fmt.Errorf("hasLock indexed at %v, got %v", clientId, index))
			}
			return matcher
		})),
		distsys.SetTraceRecorder(recorder),
	)
	defer ctx.Stop()
	go runOrPanic(ctx)

	<-matcher.AwaitValue(tla.MakeBool(true))
	body()
	<-matcher.AwaitValue(tla.MakeBool(false))
}
