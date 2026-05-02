---- MODULE TV_ClientRequest ----
\* TV for ai_spec_2's ClientRequest(n, v).

EXTENDS Naturals, FiniteSets, Sequences, TLC, Json, IOUtils

CONSTANTS Nodes, MaxTerm, MaxLogLen

VARIABLES currentTerm, votedFor, state, log, commitIndex,
          votesGranted, votesRejected, electionTimeout, heartbeatTimeout,
          leader, messages, step

S == INSTANCE etcdraft

AllWindows == ndJsonDeserialize("windows_ClientRequest.ndjson")
w == AllWindows[atoi(IOEnv.WINDOW_INDEX)]

\* Log entries are plain integers (term values) in this spec
MakeLog(len) == [i \in 1..len |-> 0]

vars == <<currentTerm, votedFor, state, log, commitIndex,
          votesGranted, votesRejected, electionTimeout, heartbeatTimeout,
          leader, messages, step>>

Init ==
    /\ currentTerm = w.pre.currentTerm
    /\ state       = w.pre.state
    /\ votedFor    = w.pre.votedFor
    /\ commitIndex = w.pre.commitIndex
    /\ log = [n \in Nodes |-> MakeLog(w.pre.logLen[n])]
    \* aux vars: plausible defaults
    /\ votesGranted = [n \in Nodes |-> {}]
    /\ votesRejected = [n \in Nodes |-> {}]
    /\ electionTimeout = [n \in Nodes |-> 0]
    /\ heartbeatTimeout = [n \in Nodes |-> 0]
    /\ leader = 1
    /\ messages = {}
    /\ step = 0

Next ==
    /\ step = 0
    /\ \E n \in Nodes, v \in 0..MaxLogLen : S!ClientRequest(n, v)
    /\ step' = 1

PostReached ==
    /\ step = 1
    /\ currentTerm = w.post.currentTerm
    /\ state = w.post.state
    /\ votedFor = w.post.votedFor
    /\ commitIndex = w.post.commitIndex
    /\ \A n \in Nodes : Len(log[n]) = w.post.logLen[n]

NeverPost == ~PostReached

Spec == Init /\ [][Next]_vars

====
