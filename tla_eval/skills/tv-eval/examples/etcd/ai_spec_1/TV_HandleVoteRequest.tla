---- MODULE TV_HandleVoteRequest ----
\* TV for HandleVoteRequest(m): a message-handling action.
\* Window's "input" field carries the triggering MsgVote message.

EXTENDS Naturals, FiniteSets, Sequences, TLC, Json, IOUtils

CONSTANTS Nodes, MaxTerm, MaxLogLen
CONSTANTS StateFollower, StateCandidate, StateLeader, StatePreCandidate
CONSTANTS MsgHup, MsgVote, MsgVoteResp, MsgPreVote, MsgPreVoteResp
CONSTANTS MsgApp, MsgAppResp, MsgHeartbeat, MsgHeartbeatResp, MsgProp, MsgBeat
CONSTANTS None

VARIABLES currentTerm, votedFor, log, state, commitIndex
VARIABLES leader, electionElapsed, heartbeatElapsed, messages
VARIABLES step

S == INSTANCE etcdraft

AllWindows == ndJsonDeserialize("windows_HandleVoteRequest.ndjson")
w == AllWindows[atoi(IOEnv.WINDOW_INDEX)]

DummyEntry == [term |-> 0, data |-> "dummy"]
MakeLog(len) == [i \in 1..len |-> DummyEntry]

\* Reconstruct the triggering MsgVote message from window's input.
\* Spec's messages have many fields; we only populate the ones HandleVoteRequest reads.
\* Other fields get placeholder values (spec's action doesn't read them).
TriggerMsg == [
    type |-> MsgVote,
    from |-> w.input.from,
    to |-> w.input.to,
    term |-> w.input.term,
    logIndex |-> w.input.logIndex,
    logTerm |-> w.input.logTerm,
    voteGranted |-> FALSE,
    prevIndex |-> 0,
    prevTerm |-> 0,
    entries |-> <<>>,
    commitIndex |-> 0,
    success |-> FALSE,
    matchIndex |-> 0
]

vars == <<currentTerm, votedFor, log, state, commitIndex,
          leader, electionElapsed, heartbeatElapsed, messages, step>>

Init ==
    /\ currentTerm = w.pre.currentTerm
    /\ state       = w.pre.state
    /\ votedFor    = w.pre.votedFor
    /\ commitIndex = w.pre.commitIndex
    /\ log = [n \in Nodes |-> MakeLog(w.pre.logLen[n])]
    /\ leader = [n \in Nodes |-> None]
    /\ electionElapsed = [n \in Nodes |-> 0]
    /\ heartbeatElapsed = [n \in Nodes |-> 0]
    /\ messages = {TriggerMsg}
    /\ step = 0

Next ==
    /\ step = 0
    /\ \E m \in messages : S!HandleVoteRequest(m)
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
