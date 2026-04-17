// TRACE-ENABLED variant of TestProducerConsumer.
// Copied into data/repositories/pgo/systems/dqueue/ by run.sh.
// Writes PGo-native NDJSON traces to the file named by DQUEUE_TRACE_FILE env var
// (or skips tracing if unset). Does not modify the upstream test.

package dqueue

import (
	"fmt"
	"os"
	"testing"

	"github.com/DistCompiler/pgo/distsys"
	"github.com/DistCompiler/pgo/distsys/resources"
	"github.com/DistCompiler/pgo/distsys/tla"
	"github.com/DistCompiler/pgo/distsys/trace"
)

func TestProducerConsumerWithTrace(t *testing.T) {
	tracePath := os.Getenv("DQUEUE_TRACE_FILE")
	if tracePath == "" {
		t.Skip("DQUEUE_TRACE_FILE not set; skipping trace capture")
	}
	_ = os.Remove(tracePath)
	f, err := os.Create(tracePath)
	if err != nil {
		t.Fatalf("cannot open trace file %s: %v", tracePath, err)
	}
	defer f.Close()
	recorder := trace.MakeLocalFileRecorder(f)

	producerSelf := tla.MakeNumber(0)
	producerInputChannel := make(chan tla.Value, 3)

	consumerSelf := tla.MakeNumber(1)
	consumerOutputChannel := make(chan tla.Value, 3)

	ctxProducer := distsys.NewMPCalContext(producerSelf, AProducer,
		distsys.DefineConstantValue("PRODUCER", producerSelf),
		distsys.EnsureArchetypeRefParam("net", resources.NewTCPMailboxes(func(index tla.Value) (resources.MailboxKind, string) {
			switch index.AsNumber() {
			case 0:
				return resources.MailboxesLocal, "localhost:8001"
			case 1:
				return resources.MailboxesRemote, "localhost:8002"
			default:
				panic(fmt.Errorf("unknown mailbox index %v", index))
			}
		})),
		distsys.EnsureArchetypeRefParam("s", resources.NewInputChan(producerInputChannel)),
		distsys.SetTraceRecorder(recorder))
	defer ctxProducer.Stop()
	go func() {
		if err := ctxProducer.Run(); err != nil {
			panic(err)
		}
	}()

	ctxConsumer := distsys.NewMPCalContext(consumerSelf, AConsumer,
		distsys.DefineConstantValue("PRODUCER", producerSelf),
		distsys.EnsureArchetypeRefParam("net", resources.NewTCPMailboxes(func(index tla.Value) (resources.MailboxKind, string) {
			switch index.AsNumber() {
			case 0:
				return resources.MailboxesRemote, "localhost:8001"
			case 1:
				return resources.MailboxesLocal, "localhost:8002"
			default:
				panic(fmt.Errorf("unknown mailbox index %v", index))
			}
		})),
		distsys.EnsureArchetypeRefParam("proc", resources.NewOutputChan(consumerOutputChannel)),
		distsys.SetTraceRecorder(recorder))
	defer ctxConsumer.Stop()
	go func() {
		if err := ctxConsumer.Run(); err != nil {
			panic(err)
		}
	}()

	produced := []tla.Value{
		tla.MakeNumber(1),
		tla.MakeNumber(2),
		tla.MakeNumber(3),
	}
	for _, v := range produced {
		producerInputChannel <- v
	}

	consumed := []tla.Value{<-consumerOutputChannel, <-consumerOutputChannel, <-consumerOutputChannel}
	close(consumerOutputChannel)

	if len(consumed) != len(produced) {
		t.Fatalf("consumed %v vs produced %v", consumed, produced)
	}
	for i := range produced {
		if !consumed[i].Equal(produced[i]) {
			t.Fatalf("consumed %v vs produced %v", consumed, produced)
		}
	}
}
