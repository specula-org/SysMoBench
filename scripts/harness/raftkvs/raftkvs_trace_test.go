// TRACE-ENABLED raftkvs 3-server safety test.
// Copied into data/repositories/pgo/systems/raftkvs/ by run.sh.
// Writes a single PGo-native NDJSON trace to $RAFTKVS_TRACE_FILE.

package raftkvs_test

import (
	"fmt"
	"log"
	"os"
	"testing"
	"time"

	"github.com/DistCompiler/pgo/distsys/trace"
	"github.com/DistCompiler/pgo/systems/raftkvs/bootstrap"
	"github.com/DistCompiler/pgo/systems/raftkvs/configs"
)

func TestSafety_ThreeServers_WithTrace(t *testing.T) {
	tracePath := os.Getenv("RAFTKVS_TRACE_FILE")
	if tracePath == "" {
		t.Skip("RAFTKVS_TRACE_FILE not set")
	}
	_ = os.Remove(tracePath)
	f, err := os.Create(tracePath)
	if err != nil {
		t.Fatalf("create trace file: %v", err)
	}
	defer f.Close()

	// Wire the shared recorder BEFORE any server/client context is built.
	// bootstrap.TraceRecorder is the global that trace_hook.go declares.
	bootstrap.TraceRecorder = trace.MakeLocalFileRecorder(f)
	defer func() { bootstrap.TraceRecorder = nil }()

	configPath := "configs/test-3-1.yaml"
	fmt.Printf("raftkvs trace test: config=%s\n", configPath)
	bootstrap.ResetClientFailureDetector()

	c, err := configs.ReadConfig(configPath)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}

	var servers []*bootstrap.Server
	for id := range c.Servers {
		s := bootstrap.NewServer(id, c, nil)
		servers = append(servers, s)
		defer func(srv *bootstrap.Server) {
			if err := srv.Close(); err != nil {
				log.Printf("close server %d: %v", srv.Id, err)
			}
		}(s)
		go func(srv *bootstrap.Server) {
			if err := srv.Run(); err != nil {
				log.Printf("server %d exit: %v", srv.Id, err)
			}
		}(s)
	}

	numRequestPairs := 3
	numRequests := numRequestPairs * 2

	log.Printf("waiting 3s for election to settle...")
	time.Sleep(3 * time.Second)

	reqCh := make(chan bootstrap.Request, numRequests)
	respCh := make(chan bootstrap.Response, numRequests)
	for clientId := range c.Clients {
		cl := bootstrap.NewClient(clientId, c)
		go func(cl *bootstrap.Client) {
			if err := cl.Run(reqCh, respCh); err != nil {
				log.Printf("client %d exit: %v", cl.Id, err)
			}
		}(cl)
		defer func(cl *bootstrap.Client) {
			if err := cl.Close(); err != nil {
				log.Printf("close client %d: %v", cl.Id, err)
			}
		}(cl)
	}

	log.Println("sending client requests")
	keys := []string{"key1", "key2", "key3"}
	for i := 0; i < numRequestPairs; i++ {
		key := keys[i%len(keys)]
		val := fmt.Sprintf("value%d", i)
		reqCh <- bootstrap.PutRequest{Key: key, Value: val}
		reqCh <- bootstrap.GetRequest{Key: key}
	}

	for i := 0; i < numRequests; i++ {
		resp := <-respCh
		log.Printf("resp: %+v", resp)
	}

	log.Printf("all %d responses received", numRequests)
}
