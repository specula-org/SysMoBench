---- MODULE etcdraft ----
EXTENDS Naturals, Sequences, FiniteSets, TLC
CONSTANTS Nodes,  \* Set of nodes
          Quorum, \* Minimum size for majority
          NoLeader, 
          CommandSet \* Set of possible commands

VARIABLES state,          \* [node -> {"Follower", "PreCandidate", "Candidate", "Leader"}]
          currentTerm,    \* [node -> term number]
          votedFor,       \* [node -> node voted for in current term (or NoLeader)]
          log,            \* [node -> sequence of [term, command]]
          commitIndex,    \* [node -> highest committed index]
          lastApplied,    \* [node -> last applied index]
          nextIndex,      \* [leader node -> [follower node -> next index to send]]
          matchIndex,     \* [leader node -> [follower node -> highest matched index]]
          messages,       \* Multiset of in-flight messages
          applied,        \* [node -> sequence of applied commands]
          uncommittedSize,\* [leader node -> size of uncommitted entries]
          leader,         \* Current leader node (or NoLeader)
          leadTransferee, \* Node being transferred leadership (or NoLeader)
          pendingReadIndexMessages \* [node -> queue of read requests]

vars == <<state, currentTerm, votedFor, log, commitIndex, lastApplied, 
          nextIndex, matchIndex, messages, applied, uncommittedSize, 
          leader, leadTransferee, pendingReadIndexMessages>>

Node == Nodes
Term == Nat
Index == Nat
Command == CommandSet
LogEntry == [term : Term, command : Command]
Log == Seq(LogEntry)
MessageType == {"MsgHup", "MsgBeat", "MsgProp", "MsgApp", "MsgAppResp", 
                "MsgVote", "MsgVoteResp", "MsgPreVote", "MsgPreVoteResp", 
                "MsgSnap", "MsgHeartbeat", "MsgHeartbeatResp", "MsgReadIndex", 
                "MsgReadIndexResp", "MsgTimeoutNow", "MsgTransferLeader", 
                "MsgForgetLeader"}

Message == [type : MessageType, from : Node, to : Node, term : Term, 
            index : Index, logTerm : Term, entries : Seq(LogEntry), 
            commit : Index, reject : BOOLEAN, rejectHint : Index, 
            context : STRING, granted : BOOLEAN]

IsLocalMsgTarget(n) == n = LocalAppendThread \/ n = LocalApplyThread
LocalAppendThread == CHOOSE n \in Node : TRUE
LocalApplyThread == CHOOSE n \in Node : n # LocalAppendThread

Init == 
    /\ state = [n \in Node |-> "Follower"]
    /\ currentTerm = [n \in Node |-> 0]
    /\ votedFor = [n \in Node |-> NoLeader]
    /\ log = [n \in Node |-> <<>>]
    /\ commitIndex = [n \in Node |-> 0]
    /\ lastApplied = [n \in Node |-> 0]
    /\ nextIndex = [n \in Node |-> [m \in Node |-> 1]]
    /\ matchIndex = [n \in Node |-> [m \in Node |-> 0]]
    /\ messages = {}
    /\ applied = [n \in Node |-> <<>>]
    /\ uncommittedSize = [n \in Node |-> 0]
    /\ leader = NoLeader
    /\ leadTransferee = NoLeader
    /\ pendingReadIndexMessages = [n \in Node |-> <<>>]

TypeInvariant == 
    /\ state \in [Node -> {"Follower", "PreCandidate", "Candidate", "Leader"}]
    /\ currentTerm \in [Node -> Term]
    /\ votedFor \in [Node -> Node \cup {NoLeader}]
    /\ log \in [Node -> Log]
    /\ commitIndex \in [Node -> Index]
    /\ lastApplied \in [Node -> Index]
    /\ nextIndex \in [Node -> [Node -> Index]]
    /\ matchIndex \in [Node -> [Node -> Index]]
    /\ messages \subseteq Message
    /\ applied \in [Node -> Seq(Command)]
    /\ uncommittedSize \in [Node -> Nat]
    /\ leader \in Node \cup {NoLeader}
    /\ leadTransferee \in Node \cup {NoLeader}
    /\ pendingReadIndexMessages \in [Node -> Seq(Message)]

\* Helper functions
LastLogIndex(n) == Len(log[n])
LastLogTerm(n) == IF LastLogIndex(n) > 0 THEN log[n][LastLogIndex(n)].term ELSE 0
LogTerm(n, i) == IF i > 0 /\ i <= Len(log[n]) THEN log[n][i].term ELSE 0
LogEntry(n, i) == IF i > 0 /\ i <= Len(log[n]) THEN log[n][i] ELSE [term |-> 0, command |-> ""]
LogSlice(n, from, to) == IF from <= to THEN SubSeq(log[n], from, to) ELSE <<>>
MajorityApproves(S) == Cardinality(S) >= Quorum
IsUpToDate(candidateTerm, candidateIndex, n) == 
    candidateTerm > LastLogTerm(n) \/ 
    (candidateTerm = LastLogTerm(n) /\ candidateIndex >= LastLogIndex(n))
IsLeader(n) == state[n] = "Leader"
IsCandidate(n) == state[n] = "Candidate" \/ state[n] = "PreCandidate"
IsFollower(n) == state[n] = "Follower"
PayloadSize(entries) == IF entries = <<>> THEN 0 ELSE 1 \* Simplified for model

\* Message constructors
MakeAppendEntries(from, to, term, prevIndex, prevTerm, entries, commit) == 
    [type |-> "MsgApp", from |-> from, to |-> to, term |-> term, 
     index |-> prevIndex, logTerm |-> prevTerm, entries |-> entries, 
     commit |-> commit, reject |-> FALSE, rejectHint |-> 0, 
     context |-> "", granted |-> FALSE]

MakeAppendResp(from, to, term, index, reject, rejectHint, logTerm) == 
    [type |-> "MsgAppResp", from |-> from, to |-> to, term |-> term, 
     index |-> index, logTerm |-> logTerm, entries |-> <<>>, 
     commit |-> 0, reject |-> reject, rejectHint |-> rejectHint, 
     context |-> "", granted |-> FALSE]

MakeVoteMsg(t, from, to, term, lastIndex, lastTerm) == 
    [type |-> t, from |-> from, to |-> to, term |-> term, 
     index |-> lastIndex, logTerm |-> lastTerm, entries |-> <<>>, 
     commit |-> 0, reject |-> FALSE, rejectHint |-> 0, 
     context |-> "", granted |-> FALSE]

MakeVoteResp(t, from, to, term, granted) == 
    [type |-> t, from |-> from, to |-> to, term |-> term, 
     index |-> 0, logTerm |-> 0, entries |-> <<>>, 
     commit |-> 0, reject |-> ~granted, rejectHint |-> 0, 
     context |-> "", granted |-> granted]

\* Election timeout handler
BecomePreCandidate(n) == 
    /\ state[n] \in {"Follower", "PreCandidate"}
    /\ state' = [state EXCEPT ![n] = "PreCandidate"]
    /\ \E newTerm \in Term: 
        /\ newTerm = currentTerm[n] + 1
        /\ \A m \in Node:
            LET voteMsg == MakeVoteMsg("MsgPreVote", n, m, newTerm, LastLogIndex(n), LastLogTerm(n)
            IN messages' = messages \cup {voteMsg}
    /\ UNCHANGED <<currentTerm, votedFor, log, commitIndex, lastApplied, nextIndex, 
                   matchIndex, applied, uncommittedSize, leader, leadTransferee, 
                   pendingReadIndexMessages>>

BecomeCandidate(n) == 
    /\ state[n] \in {"Follower", "Candidate", "PreCandidate"}
    /\ currentTerm' = [currentTerm EXCEPT ![n] = currentTerm[n] + 1]
    /\ state' = [state EXCEPT ![n] = "Candidate"]
    /\ votedFor' = [votedFor EXCEPT ![n] = n]
    /\ \A m \in Node:
        LET voteMsg == MakeVoteMsg("MsgVote", n, m, currentTerm'[n], LastLogIndex(n), LastLogTerm(n))
        IN messages' = messages \cup {voteMsg}
    /\ UNCHANGED <<log, commitIndex, lastApplied, nextIndex, matchIndex, applied, 
                   uncommittedSize, leader, leadTransferee, pendingReadIndexMessages>>

BecomeLeader(n) == 
    /\ state[n] = "Candidate"
    /\ state' = [state EXCEPT ![n] = "Leader"]
    /\ leader' = n
    /\ nextIndex' = [nextIndex EXCEPT ![n] = [m \in Node |-> LastLogIndex(n) + 1]]
    /\ matchIndex' = [matchIndex EXCEPT ![n] = [m \in Node |-> 0]]
    /\ uncommittedSize' = [uncommittedSize EXCEPT ![n] = 0]
    /\ \E emptyEntry \in LogEntry: 
        /\ log' = [log EXCEPT ![n] = Append(log[n], [term |-> currentTerm[n], command |-> emptyEntry.command])]
        /\ \A m \in Node:
            LET aeMsg == MakeAppendEntries(n, m, currentTerm[n], LastLogIndex(n) - 1, 
                            LogTerm(n, LastLogIndex(n) - 1), <<>>, commitIndex[n])
            IN messages' = messages \cup {aeMsg}
    /\ UNCHANGED <<currentTerm, votedFor, commitIndex, lastApplied, applied, 
                   leadTransferee, pendingReadIndexMessages>>

BecomeFollower(n, term, lead) == 
    /\ state' = [state EXCEPT ![n] = "Follower"]
    /\ currentTerm' = [currentTerm EXCEPT ![n] = term]
    /\ votedFor' = [votedFor EXCEPT ![n] = NoLeader]
    /\ leader' = lead
    /\ leadTransferee' = NoLeader
    /\ UNCHANGED <<log, commitIndex, lastApplied, nextIndex, matchIndex, applied, 
                   uncommittedSize, pendingReadIndexMessages>>

\* Vote handlers
HandleRequestVote(n, m) == 
    LET grant == 
        /\ m.term >= currentTerm[n]
        /\ (votedFor[n] = NoLeader \/ votedFor[n] = m.from)
        /\ IsUpToDate(m.logTerm, m.index, n)
    IN
    /\ IF grant 
        THEN votedFor' = [votedFor EXCEPT ![n] = m.from]
        ELSE UNCHANGED votedFor
    /\ messages' = messages \cup {MakeVoteResp("MsgVoteResp", n, m.from, currentTerm[n], grant)}
    /\ UNCHANGED <<state, currentTerm, log, commitIndex, lastApplied, nextIndex, 
                   matchIndex, applied, uncommittedSize, leader, leadTransferee, 
                   pendingReadIndexMessages>>

HandlePreVote(n, m) == 
    LET grant == 
        /\ m.term > currentTerm[n]
        /\ (votedFor[n] = NoLeader \/ votedFor[n] = m.from)
        /\ IsUpToDate(m.logTerm, m.index, n)
    IN
    /\ messages' = messages \cup {MakeVoteResp("MsgPreVoteResp", n, m.from, currentTerm[n], grant)}
    /\ UNCHANGED <<state, currentTerm, votedFor, log, commitIndex, lastApplied, 
                   nextIndex, matchIndex, applied, uncommittedSize, leader, 
                   leadTransferee, pendingReadIndexMessages>>

\* AppendEntries handlers
HandleAppendEntries(n, m) == 
    LET prevOk == 
        IF m.index = 0 THEN TRUE 
        ELSE m.index <= LastLogIndex(n) /\ LogTerm(n, m.index) = m.logTerm
    IN
    IF ~prevOk \/ m.term < currentTerm[n] 
    THEN 
        /\ messages' = messages \cup {MakeAppendResp(n, m.from, currentTerm[n], m.index, TRUE, 
                            LastLogIndex(n), LastLogTerm(n))}
        /\ UNCHANGED <<state, currentTerm, votedFor, log, commitIndex, lastApplied, 
                       nextIndex, matchIndex, applied, uncommittedSize, leader, 
                       leadTransferee, pendingReadIndexMessages>>
    ELSE 
        /\ IF m.term > currentTerm[n] 
            THEN BecomeFollower(n, m.term, m.from)
            ELSE UNCHANGED <<state, currentTerm, votedFor, leader>>
        /\ \* Truncate log if conflict
        LET conflict == 
            IF m.entries # <<>> THEN
                \E i \in 1..Min(Len(log[n]) - m.index, Len(m.entries)) : 
                    log[n][m.index + i].term # m.entries[i].term
            ELSE FALSE
        IN
        LET newLog == 
            IF conflict THEN 
                SubSeq(log[n], 1, m.index) \o m.entries
            ELSE 
                log[n] \o SubSeq(m.entries, Len(log[n]) - m.index + 1, Len(m.entries))
        IN
        /\ log' = [log EXCEPT ![n] = newLog]
        /\ commitIndex' = [commitIndex EXCEPT ![n] = Min(m.commit, LastLogIndex(n))]
        /\ messages' = messages \cup {MakeAppendResp(n, m.from, currentTerm[n], LastLogIndex(n), FALSE, 0, 0)}
        /\ UNCHANGED <<nextIndex, matchIndex, applied, uncommittedSize, leadTransferee, 
                       pendingReadIndexMessages, lastApplied>>

\* Leader actions
LeaderAppend(n, cmd) == 
    /\ IsLeader(n)
    /\ \E newEntry \in LogEntry: 
        /\ newEntry.term = currentTerm[n]
        /\ newEntry.command = cmd
        /\ log' = [log EXCEPT ![n] = Append(log[n], newEntry)]
        /\ uncommittedSize' = [uncommittedSize EXCEPT ![n] = uncommittedSize[n] + PayloadSize(<<newEntry>>)]
        /\ \A m \in Node:
            LET aeMsg == MakeAppendEntries(n, m, currentTerm[n], LastLogIndex(n) - 1, 
                            LogTerm(n, LastLogIndex(n) - 1), <<newEntry>>, commitIndex[n])
            IN messages' = messages \cup {aeMsg}
        /\ UNCHANGED <<state, currentTerm, votedFor, commitIndex, lastApplied, nextIndex, 
                       matchIndex, applied, leader, leadTransferee, pendingReadIndexMessages>>

LeaderCommit(n) == 
    /\ IsLeader(n)
    /\ \E N \in Index: 
        /\ N > commitIndex[n]
        /\ Cardinality({m \in Node : matchIndex[n][m] >= N}) >= Quorum
        /\ LogTerm(n, N) = currentTerm[n]
        /\ commitIndex' = [commitIndex EXCEPT ![n] = N]
        /\ \A k \in Node: 
            IF k # n 
            THEN messages' = messages \cup {MakeAppendEntries(n, k, currentTerm[n], 
                                    LastLogIndex(n) - 1, LogTerm(n, LastLogIndex(n) - 1), 
                                    <<>>, N)}
            ELSE UNCHANGED
    /\ UNCHANGED <<state, currentTerm, votedFor, log, lastApplied, nextIndex, 
                   matchIndex, applied, uncommittedSize, leader, leadTransferee, 
                   pendingReadIndexMessages>>

ApplyLogEntry(n) == 
    /\ lastApplied[n] < commitIndex[n]
    /\ LET idx == lastApplied[n] + 1
        entry == LogEntry(n, idx)
        applied' = [applied EXCEPT ![n] = Append(applied[n], entry.command)]
        lastApplied' = [lastApplied EXCEPT ![n] = idx]
    IN
    /\ UNCHANGED <<state, currentTerm, votedFor, log, commitIndex, nextIndex, 
                   matchIndex, uncommittedSize, leader, leadTransferee, 
                   pendingReadIndexMessages, messages>>

\* ReadIndex handling
HandleReadIndex(n, m) == 
    /\ IsLeader(n)
    /\ pendingReadIndexMessages' = [pendingReadIndexMessages EXCEPT ![n] = Append(pendingReadIndexMessages[n], m)]
    /\ UNCHANGED <<state, currentTerm, votedFor, log, commitIndex, lastApplied, 
                   nextIndex, matchIndex, applied, uncommittedSize, leader, 
                   leadTransferee, messages>>

ReleaseReadIndex(n) == 
    /\ IsLeader(n)
    /\ commitIndex[n] > 0
    /\ \E reqs \in Seq(Message): 
        /\ pendingReadIndexMessages[n] # <<>>
        /\ pendingReadIndexMessages' = [pendingReadIndexMessages EXCEPT ![n] = <<>>]
        /\ \A msg \in reqs: 
            LET resp == [type |-> "MsgReadIndexResp", from |-> n, to |-> msg.from, 
                         term |-> currentTerm[n], index |-> commitIndex[n], 
                         entries |-> msg.entries, context |-> ""]
            IN messages' = messages \cup {resp}
    /\ UNCHANGED <<state, currentTerm, votedFor, log, commitIndex, lastApplied, 
                   nextIndex, matchIndex, applied, uncommittedSize, leader, 
                   leadTransferee>>

\* Next-state relation
Next == 
    \/ \E n \in Node: \/ BecomePreCandidate(n)
                      \/ BecomeCandidate(n)
                      \/ BecomeLeader(n)
                      \/ ApplyLogEntry(n)
                      \/ HandleReadIndex(n, CHOOSE m \in messages: m.type = "MsgReadIndex" /\ m.to = n)
                      \/ ReleaseReadIndex(n)
    \/ \E m \in messages: 
        \/ \E n \in Node: 
            \/ (m.to = n /\ m.type = "MsgVote") /\ HandleRequestVote(n, m)
            \/ (m.to = n /\ m.type = "MsgPreVote") /\ HandlePreVote(n, m)
            \/ (m.to = n /\ m.type = "MsgApp") /\ HandleAppendEntries(n, m)
            \/ (m.to = n /\ m.type = "MsgProp") /\ LeaderAppend(n, CHOOSE cmd \in Command: TRUE)
            \/ (m.to = n /\ m.type = "MsgAppResp" /\ IsLeader(m.from)) /\ LeaderCommit(m.from)
    \/ \E n \in Node, cmd \in Command: LeaderAppend(n, cmd)

\* Safety properties
Safety == 
    \* Election safety: At most one leader per term
    \A t \in Term: 
        Cardinality({n \in Node: state[n] = "Leader" /\ currentTerm[n] = t}) <= 1

    \* Log matching: If logs have same term/index, they are identical
    \A n1, n2 \in Node, i \in 1..Min(LastLogIndex(n1), LastLogIndex(n2)):
        LogTerm(n1, i) = LogTerm(n2, i) => LogEntry(n1, i) = LogEntry(n2, i)

    \* State machine safety: Applied commands are same across nodes
    \A n1, n2 \in Node, i \in 1..Min(Len(applied[n1]), Len(applied[n2])):
        applied[n1][i] = applied[n2][i]

\* Temporal properties
Liveness == 
    <>[]( \E n \in Node: state[n] = "Leader" ) \* Eventually always a leader

\* Specification
Spec == Init /\ [][Next]_vars

====