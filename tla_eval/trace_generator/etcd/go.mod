module github.com/etcd-io/tla-benchmark/trace-generator

go 1.24

toolchain go1.24.5

// Use the local etcd raft repository
replace go.etcd.io/raft/v3 => ../../../data/repositories/raft

require go.etcd.io/raft/v3 v3.0.0-00010101000000-000000000000

require (
	github.com/cockroachdb/datadriven v1.0.2 // indirect
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/gogo/protobuf v1.3.2 // indirect
	github.com/golang/protobuf v1.5.4 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/stretchr/testify v1.10.0 // indirect
	google.golang.org/protobuf v1.33.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
