---- MODULE TV_HandleAppendEntries ----
\* TV for HandleAppendEntries(m): log-replication message handler.
\* Uses logLen + logLastTerm abstraction:
\*   - Init log as a sequence of dummy entries with correct length
\*   - Last entry's term = w.pre.logLastTerm[n]
\*   - Middle entries have term 0 (doesn't matter for prevLogTerm check
\*     since prevLogTerm compares against Len(log[n])-th entry)

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

AllWindows == ndJsonDeserialize("windows_HandleAppendEntries.ndjson")
w == AllWindows[atoi(IOEnv.WINDOW_INDEX)]

DummyEntry == [term |-> 0, data |-> "dummy"]

\* Make a log of given length where the last entry has the given term.
\* Preserves logLastTerm faithfully; middle entries stay as DummyEntry.
MakeLog(len, lastTerm) ==
    IF len = 0 THEN <<>>
    ELSE [i \in 1..len |-> IF i = len THEN [term |-> lastTerm, data |-> "dummy"]
                           ELSE DummyEntry]

\* Reconstruct the triggering MsgApp message from window's input.
EntryFromInput(e) == [term |-> e.term, data |-> "dummy"]
InputEntries == [i \in 1..Len(w.input.entries) |-> EntryFromInput(w.input.entries[i])]

TriggerMsg == [
    type |-> MsgApp,
    from |-> w.input.from,
    to |-> w.input.to,
    term |-> w.input.term,
    prevIndex |-> w.input.prevLogIndex,
    prevTerm |-> w.input.prevLogTerm,
    entries |-> InputEntries,
    commitIndex |-> w.input.commitIndex,
    \* pad with other fields the spec's message record expects
    logIndex |-> 0,
    logTerm |-> 0,
    voteGranted |-> FALSE,
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
    /\ log = [n \in Nodes |-> MakeLog(w.pre.logLen[n], w.pre.logLastTerm[n])]
    /\ leader = [n \in Nodes |-> None]
    /\ electionElapsed = [n \in Nodes |-> 0]
    /\ heartbeatElapsed = [n \in Nodes |-> 0]
    /\ messages = {TriggerMsg}
    /\ step = 0

Next ==
    /\ step = 0
    /\ \E m \in messages : S!HandleAppendEntries(m)
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
