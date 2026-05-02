---- MODULE TV_HandleVoteRequest ----
\* TV for ai_spec_2's HandleVoteRequest(n, m).

EXTENDS Naturals, FiniteSets, Sequences, TLC, Json, IOUtils

CONSTANTS Nodes, MaxTerm, MaxLogLen

VARIABLES currentTerm, votedFor, state, log, commitIndex,
          votesGranted, votesRejected, electionTimeout, heartbeatTimeout,
          leader, messages, step

S == INSTANCE etcdraft

AllWindows == ndJsonDeserialize("windows_HandleVoteRequest.ndjson")
w == AllWindows[atoi(IOEnv.WINDOW_INDEX)]

MakeLog(len) == [i \in 1..len |-> 0]

\* Build the triggering MsgVote. This spec uses 'granted' not 'voteGranted'.
\* Include all fields the spec's HandleVoteRequest and Reply reference.
TriggerMsg == [
    type |-> "MsgVote",
    from |-> w.input.from,
    to |-> w.input.to,
    term |-> w.input.term,
    logIndex |-> w.input.logIndex,
    logTerm |-> w.input.logTerm
]

vars == <<currentTerm, votedFor, state, log, commitIndex,
          votesGranted, votesRejected, electionTimeout, heartbeatTimeout,
          leader, messages, step>>

Init ==
    /\ currentTerm = w.pre.currentTerm
    /\ state       = w.pre.state
    /\ votedFor    = w.pre.votedFor
    /\ commitIndex = w.pre.commitIndex
    /\ log = [n \in Nodes |-> MakeLog(w.pre.logLen[n])]
    /\ votesGranted = [n \in Nodes |-> {}]
    /\ votesRejected = [n \in Nodes |-> {}]
    /\ electionTimeout = [n \in Nodes |-> 0]
    /\ heartbeatTimeout = [n \in Nodes |-> 0]
    /\ leader = 1
    /\ messages = {TriggerMsg}
    /\ step = 0

Next ==
    /\ step = 0
    /\ \E n \in Nodes, m \in messages : S!HandleVoteRequest(n, m)
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
