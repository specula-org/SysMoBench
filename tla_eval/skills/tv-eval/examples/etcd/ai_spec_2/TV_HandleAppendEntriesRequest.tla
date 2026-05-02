---- MODULE TV_HandleAppendEntriesRequest ----
\* TV for ai_spec_2's HandleAppendEntriesRequest(n, m).
\* Log entries are integers (term values) in this spec.

EXTENDS Naturals, FiniteSets, Sequences, TLC, Json, IOUtils

CONSTANTS Nodes, MaxTerm, MaxLogLen

VARIABLES currentTerm, votedFor, state, log, commitIndex,
          votesGranted, votesRejected, electionTimeout, heartbeatTimeout,
          leader, messages, step

S == INSTANCE etcdraft

AllWindows == ndJsonDeserialize("windows_HandleAppendEntriesRequest.ndjson")
w == AllWindows[atoi(IOEnv.WINDOW_INDEX)]

\* Log entry = term integer. Last entry = logLastTerm, others = 0.
MakeLog(len, lastTerm) ==
    IF len = 0 THEN <<>>
    ELSE [i \in 1..len |-> IF i = len THEN lastTerm ELSE 0]

\* Entries in the input message: each is an integer term
InputEntries == [i \in 1..Len(w.input.entries) |-> w.input.entries[i].term]

TriggerMsg == [
    type |-> "MsgApp",
    from |-> w.input.from,
    to |-> w.input.to,
    term |-> w.input.term,
    prevLogIndex |-> w.input.prevLogIndex,
    prevLogTerm |-> w.input.prevLogTerm,
    entries |-> InputEntries,
    commitIndex |-> w.input.commitIndex
]

vars == <<currentTerm, votedFor, state, log, commitIndex,
          votesGranted, votesRejected, electionTimeout, heartbeatTimeout,
          leader, messages, step>>

Init ==
    /\ currentTerm = w.pre.currentTerm
    /\ state       = w.pre.state
    /\ votedFor    = w.pre.votedFor
    /\ commitIndex = w.pre.commitIndex
    /\ log = [n \in Nodes |-> MakeLog(w.pre.logLen[n], w.pre.logLastTerm[n])]
    /\ votesGranted = [n \in Nodes |-> {}]
    /\ votesRejected = [n \in Nodes |-> {}]
    /\ electionTimeout = [n \in Nodes |-> 0]
    /\ heartbeatTimeout = [n \in Nodes |-> 0]
    /\ leader = 1
    /\ messages = {TriggerMsg}
    /\ step = 0

Next ==
    /\ step = 0
    /\ \E n \in Nodes, m \in messages : S!HandleAppendEntriesRequest(n, m)
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
