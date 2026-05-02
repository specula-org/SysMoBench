---- MODULE TV_ClientProposal ----
\* Window Validator for ClientProposal. Reads windows from JSON at runtime.

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

AllWindows == ndJsonDeserialize("windows_ClientProposal.ndjson")
w == AllWindows[atoi(IOEnv.WINDOW_INDEX)]

DummyEntry == [term |-> 0, data |-> "dummy"]
MakeLog(len) == [i \in 1..len |-> DummyEntry]

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
    /\ messages = {}
    /\ step = 0

Next ==
    /\ step = 0
    /\ \E n \in Nodes : S!ClientProposal(n)
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
