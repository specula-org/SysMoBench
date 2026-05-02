---- MODULE etcdraft ----
EXTENDS TLC, Sequences, SequencesExt, Naturals, FiniteSets, Bags

CONSTANTS
    Nodes,
    MaxTerm,
    MaxLogLen

VARIABLES
    currentTerm,
    votedFor,
    state,
    log,
    commitIndex,
    votesGranted,
    votesRejected,
    electionTimeout,
    heartbeatTimeout,
    leader,
    messages

vars == <<currentTerm, votedFor, state, log, commitIndex, votesGranted,
          votesRejected, electionTimeout, heartbeatTimeout, leader, messages>>

StateFollower == "Follower"
StatePreCandidate == "PreCandidate"
StateCandidate == "Candidate"
StateLeader == "Leader"

MsgHup == "MsgHup"
MsgPreVote == "MsgPreVote"
MsgPreVoteResp == "MsgPreVoteResp"
MsgVote == "MsgVote"
MsgVoteResp == "MsgVoteResp"
MsgApp == "MsgApp"
MsgAppResp == "MsgAppResp"
MsgHeartbeat == "MsgHeartbeat"
MsgProp == "MsgProp"

Message ==
    [type: {MsgPreVote}, from: Nodes, to: Nodes, term: Nat, logIndex: Nat, logTerm: Nat]
    \union [type: {MsgPreVoteResp}, from: Nodes, to: Nodes, term: Nat, granted: BOOLEAN]
    \union [type: {MsgVote}, from: Nodes, to: Nodes, term: Nat, logIndex: Nat, logTerm: Nat]
    \union [type: {MsgVoteResp}, from: Nodes, to: Nodes, term: Nat, granted: BOOLEAN]
    \union [type: {MsgApp}, from: Nodes, to: Nodes, term: Nat, prevLogIndex: Nat,
            prevLogTerm: Nat, entries: Seq(Nat), commitIndex: Nat]
    \union [type: {MsgAppResp}, from: Nodes, to: Nodes, term: Nat, success: BOOLEAN, matchIndex: Nat]
    \union [type: {MsgHeartbeat}, from: Nodes, to: Nodes, term: Nat, commitIndex: Nat]
    \union [type: {MsgProp}, from: Nodes, to: Nodes, value: Nat]

Quorum == {S \in SUBSET Nodes : Cardinality(S) * 2 > Cardinality(Nodes)}

Init ==
    /\ currentTerm = [n \in Nodes |-> 0]
    /\ votedFor = [n \in Nodes |-> CHOOSE x \in Nodes : TRUE]
    /\ state = [n \in Nodes |-> StateFollower]
    /\ log = [n \in Nodes |-> <<>>]
    /\ commitIndex = [n \in Nodes |-> 0]
    /\ votesGranted = [n \in Nodes |-> {}]
    /\ votesRejected = [n \in Nodes |-> {}]
    /\ electionTimeout = [n \in Nodes |-> TRUE]
    /\ heartbeatTimeout = [n \in Nodes |-> FALSE]
    /\ leader = [n \in Nodes |-> CHOOSE x \in Nodes : TRUE]
    /\ messages = {}

Send(m) == messages' = messages \union {m}

Discard(m) == messages' = messages \ {m}

Reply(response, request) ==
    messages' = (messages \ {request}) \union {response}

GetLastLogIndex(n) == Len(log[n])

GetLastLogTerm(n) == IF Len(log[n]) = 0 THEN 0 ELSE log[n][Len(log[n])]

LogOk(n, m) ==
    \/ m.prevLogIndex = 0
    \/ /\ m.prevLogIndex > 0
       /\ m.prevLogIndex <= Len(log[n])
       /\ log[n][m.prevLogIndex] = m.prevLogTerm

IsUpToDate(n, candIndex, candTerm) ==
    LET lastTerm == GetLastLogTerm(n)
        lastIndex == GetLastLogIndex(n)
    IN \/ candTerm > lastTerm
       \/ /\ candTerm = lastTerm
          /\ candIndex >= lastIndex

BecomeFollower(n, term) ==
    /\ state' = [state EXCEPT ![n] = StateFollower]
    /\ currentTerm' = [currentTerm EXCEPT ![n] = term]
    /\ votedFor' = [votedFor EXCEPT ![n] = CHOOSE x \in Nodes : TRUE]
    /\ leader' = [leader EXCEPT ![n] = CHOOSE x \in Nodes : TRUE]
    /\ votesGranted' = [votesGranted EXCEPT ![n] = {}]
    /\ votesRejected' = [votesRejected EXCEPT ![n] = {}]

BecomePreCandidate(n) ==
    /\ state[n] # StateLeader
    /\ state' = [state EXCEPT ![n] = StatePreCandidate]
    /\ leader' = [leader EXCEPT ![n] = CHOOSE x \in Nodes : TRUE]
    /\ votesGranted' = [votesGranted EXCEPT ![n] = {}]
    /\ votesRejected' = [votesRejected EXCEPT ![n] = {}]
    /\ UNCHANGED <<currentTerm, votedFor>>

BecomeCandidate(n) ==
    /\ state[n] # StateLeader
    /\ state' = [state EXCEPT ![n] = StateCandidate]
    /\ currentTerm' = [currentTerm EXCEPT ![n] = currentTerm[n] + 1]
    /\ votedFor' = [votedFor EXCEPT ![n] = n]
    /\ leader' = [leader EXCEPT ![n] = CHOOSE x \in Nodes : TRUE]
    /\ votesGranted' = [votesGranted EXCEPT ![n] = {}]
    /\ votesRejected' = [votesRejected EXCEPT ![n] = {}]

BecomeLeader(n) ==
    /\ state[n] \in {StateCandidate}
    /\ state' = [state EXCEPT ![n] = StateLeader]
    /\ leader' = [leader EXCEPT ![n] = n]

Timeout(n) ==
    /\ state[n] \in {StateFollower, StatePreCandidate, StateCandidate}
    /\ electionTimeout[n]
    /\ BecomePreCandidate(n)
    /\ LET lastIndex == GetLastLogIndex(n)
           lastTerm == GetLastLogTerm(n)
           preVoteMsg == [type |-> MsgPreVote, from |-> n, to |-> CHOOSE x \in Nodes : TRUE,
                          term |-> currentTerm[n] + 1, logIndex |-> lastIndex, logTerm |-> lastTerm]
       IN messages' = messages \union {[preVoteMsg EXCEPT !.to = m] : m \in Nodes \ {n}}
    /\ UNCHANGED <<log, commitIndex, electionTimeout, heartbeatTimeout>>

RequestPreVote(n) ==
    /\ state[n] = StatePreCandidate
    /\ currentTerm[n] < MaxTerm
    /\ LET lastIndex == GetLastLogIndex(n)
           lastTerm == GetLastLogTerm(n)
           preVoteMsg == [type |-> MsgPreVote, from |-> n, to |-> CHOOSE x \in Nodes : TRUE,
                          term |-> currentTerm[n] + 1, logIndex |-> lastIndex, logTerm |-> lastTerm]
       IN messages' = messages \union {[preVoteMsg EXCEPT !.to = m] : m \in Nodes \ {n}}
    /\ UNCHANGED <<currentTerm, votedFor, state, log, commitIndex, votesGranted,
                   votesRejected, electionTimeout, heartbeatTimeout, leader>>

HandlePreVoteRequest(n, m) ==
    /\ m.type = MsgPreVote
    /\ m.to = n
    /\ LET grant == IsUpToDate(n, m.logIndex, m.logTerm)
           response == [type |-> MsgPreVoteResp, from |-> n, to |-> m.from,
                        term |-> m.term, granted |-> grant]
       IN Reply(response, m)
    /\ UNCHANGED <<currentTerm, votedFor, state, log, commitIndex, votesGranted,
                   votesRejected, electionTimeout, heartbeatTimeout, leader>>

HandlePreVoteResponse(n, m) ==
    /\ m.type = MsgPreVoteResp
    /\ m.to = n
    /\ state[n] = StatePreCandidate
    /\ m.term = currentTerm[n] + 1
    /\ \/ /\ m.granted
          /\ votesGranted' = [votesGranted EXCEPT ![n] = @ \union {m.from}]
          /\ UNCHANGED votesRejected
       \/ /\ ~m.granted
          /\ votesRejected' = [votesRejected EXCEPT ![n] = @ \union {m.from}]
          /\ UNCHANGED votesGranted
    /\ Discard(m)
    /\ UNCHANGED <<currentTerm, votedFor, state, log, commitIndex,
                   electionTimeout, heartbeatTimeout, leader>>

RequestVote(n) ==
    /\ state[n] = StatePreCandidate
    /\ votesGranted[n] \in Quorum
    /\ BecomeCandidate(n)
    /\ LET lastIndex == GetLastLogIndex(n)
           lastTerm == GetLastLogTerm(n)
           voteMsg == [type |-> MsgVote, from |-> n, to |-> CHOOSE x \in Nodes : TRUE,
                       term |-> currentTerm'[n], logIndex |-> lastIndex, logTerm |-> lastTerm]
       IN messages' = messages \union {[voteMsg EXCEPT !.to = m] : m \in Nodes \ {n}}
    /\ UNCHANGED <<log, commitIndex, electionTimeout, heartbeatTimeout>>

HandleVoteRequest(n, m) ==
    /\ m.type = MsgVote
    /\ m.to = n
    /\ LET logOk == IsUpToDate(n, m.logIndex, m.logTerm)
           grant == /\ m.term >= currentTerm[n]
                    /\ logOk
                    /\ \/ votedFor[n] = m.from
                       \/ /\ m.term > currentTerm[n]
                          /\ votedFor[n] = CHOOSE x \in Nodes : TRUE
           newTerm == IF m.term > currentTerm[n] THEN m.term ELSE currentTerm[n]
           response == [type |-> MsgVoteResp, from |-> n, to |-> m.from,
                        term |-> m.term, granted |-> grant]
       IN /\ Reply(response, m)
          /\ IF m.term > currentTerm[n]
             THEN /\ currentTerm' = [currentTerm EXCEPT ![n] = m.term]
                  /\ votedFor' = [votedFor EXCEPT ![n] = IF grant THEN m.from
                                                          ELSE CHOOSE x \in Nodes : TRUE]
                  /\ IF state[n] # StateFollower
                     THEN state' = [state EXCEPT ![n] = StateFollower]
                     ELSE UNCHANGED state
             ELSE /\ IF grant THEN votedFor' = [votedFor EXCEPT ![n] = m.from]
                               ELSE UNCHANGED votedFor
                  /\ UNCHANGED <<currentTerm, state>>
    /\ UNCHANGED <<log, commitIndex, votesGranted, votesRejected,
                   electionTimeout, heartbeatTimeout, leader>>

HandleVoteResponse(n, m) ==
    /\ m.type = MsgVoteResp
    /\ m.to = n
    /\ state[n] = StateCandidate
    /\ m.term = currentTerm[n]
    /\ \/ /\ m.granted
          /\ votesGranted' = [votesGranted EXCEPT ![n] = @ \union {m.from}]
          /\ UNCHANGED votesRejected
       \/ /\ ~m.granted
          /\ votesRejected' = [votesRejected EXCEPT ![n] = @ \union {m.from}]
          /\ UNCHANGED votesGranted
    /\ Discard(m)
    /\ UNCHANGED <<currentTerm, votedFor, state, log, commitIndex,
                   electionTimeout, heartbeatTimeout, leader>>

WinElection(n) ==
    /\ state[n] = StateCandidate
    /\ votesGranted[n] \in Quorum
    /\ BecomeLeader(n)
    /\ LET entry == currentTerm[n]
       IN /\ log' = [log EXCEPT ![n] = Append(@, entry)]
          /\ LET heartbeatMsg == [type |-> MsgHeartbeat, from |-> n, to |-> CHOOSE x \in Nodes : TRUE,
                                  term |-> currentTerm[n], commitIndex |-> commitIndex[n]]
             IN messages' = messages \union {[heartbeatMsg EXCEPT !.to = m] : m \in Nodes \ {n}}
    /\ UNCHANGED <<currentTerm, votedFor, commitIndex, votesGranted, votesRejected,
                   electionTimeout, heartbeatTimeout>>

ClientRequest(n, v) ==
    /\ state[n] = StateLeader
    /\ Len(log[n]) < MaxLogLen
    /\ LET entry == currentTerm[n]
       IN log' = [log EXCEPT ![n] = Append(@, entry)]
    /\ UNCHANGED <<currentTerm, votedFor, state, commitIndex, votesGranted,
                   votesRejected, electionTimeout, heartbeatTimeout, leader, messages>>

SendAppendEntries(n, m) ==
    /\ state[n] = StateLeader
    /\ m \in Nodes
    /\ m # n
    /\ LET prevIndex == Len(log[m])
           prevTerm == IF prevIndex = 0 THEN 0 ELSE
                       IF prevIndex <= Len(log[n]) THEN log[n][prevIndex] ELSE 0
           entries == IF prevIndex < Len(log[n])
                      THEN SubSeq(log[n], prevIndex + 1, Len(log[n]))
                      ELSE <<>>
           msg == [type |-> MsgApp, from |-> n, to |-> m, term |-> currentTerm[n],
                   prevLogIndex |-> prevIndex, prevLogTerm |-> prevTerm,
                   entries |-> entries, commitIndex |-> commitIndex[n]]
       IN Send(msg)
    /\ UNCHANGED <<currentTerm, votedFor, state, log, commitIndex, votesGranted,
                   votesRejected, electionTimeout, heartbeatTimeout, leader>>

HandleAppendEntriesRequest(n, m) ==
    /\ m.type = MsgApp
    /\ m.to = n
    /\ LET logOk == LogOk(n, m)
       IN \/ /\ m.term < currentTerm[n]
             /\ LET response == [type |-> MsgAppResp, from |-> n, to |-> m.from,
                                 term |-> currentTerm[n], success |-> FALSE, matchIndex |-> 0]
                IN Reply(response, m)
             /\ UNCHANGED <<currentTerm, votedFor, state, log, commitIndex, votesGranted,
                            votesRejected, electionTimeout, heartbeatTimeout, leader>>
          \/ /\ m.term >= currentTerm[n]
             /\ IF state[n] # StateFollower \/ currentTerm[n] # m.term
                THEN /\ currentTerm' = [currentTerm EXCEPT ![n] = m.term]
                     /\ state' = [state EXCEPT ![n] = StateFollower]
                     /\ votedFor' = [votedFor EXCEPT ![n] = CHOOSE x \in Nodes : TRUE]
                     /\ leader' = [leader EXCEPT ![n] = m.from]
                ELSE UNCHANGED <<currentTerm, votedFor, state, leader>>
             /\ \/ /\ logOk
                   /\ LET newLog == IF Len(m.entries) > 0
                                    THEN [i \in 1..(m.prevLogIndex + Len(m.entries)) |->
                                          IF i <= m.prevLogIndex THEN log[n][i]
                                          ELSE m.entries[i - m.prevLogIndex]]
                                    ELSE log[n]
                          newCommit == IF m.commitIndex > commitIndex[n]
                                       THEN m.commitIndex
                                       ELSE commitIndex[n]
                          response == [type |-> MsgAppResp, from |-> n, to |-> m.from,
                                       term |-> m.term, success |-> TRUE,
                                       matchIndex |-> m.prevLogIndex + Len(m.entries)]
                      IN /\ log' = [log EXCEPT ![n] = newLog]
                         /\ commitIndex' = [commitIndex EXCEPT ![n] = newCommit]
                         /\ Reply(response, m)
                \/ /\ ~logOk
                   /\ LET response == [type |-> MsgAppResp, from |-> n, to |-> m.from,
                                       term |-> currentTerm'[n], success |-> FALSE, matchIndex |-> 0]
                      IN /\ Reply(response, m)
                         /\ UNCHANGED <<log, commitIndex>>
             /\ UNCHANGED <<votesGranted, votesRejected, electionTimeout, heartbeatTimeout>>

HandleAppendEntriesResponse(n, m) ==
    /\ m.type = MsgAppResp
    /\ m.to = n
    /\ state[n] = StateLeader
    /\ m.term = currentTerm[n]
    /\ m.success
    /\ Discard(m)
    /\ UNCHANGED <<currentTerm, votedFor, state, log, commitIndex, votesGranted,
                   votesRejected, electionTimeout, heartbeatTimeout, leader>>

SendHeartbeat(n) ==
    /\ state[n] = StateLeader
    /\ heartbeatTimeout[n]
    /\ LET heartbeatMsg == [type |-> MsgHeartbeat, from |-> n, to |-> CHOOSE x \in Nodes : TRUE,
                            term |-> currentTerm[n], commitIndex |-> commitIndex[n]]
       IN messages' = messages \union {[heartbeatMsg EXCEPT !.to = m] : m \in Nodes \ {n}}
    /\ heartbeatTimeout' = [heartbeatTimeout EXCEPT ![n] = FALSE]
    /\ UNCHANGED <<currentTerm, votedFor, state, log, commitIndex, votesGranted,
                   votesRejected, electionTimeout, leader>>

HandleHeartbeat(n, m) ==
    /\ m.type = MsgHeartbeat
    /\ m.to = n
    /\ \/ /\ m.term < currentTerm[n]
          /\ Discard(m)
          /\ UNCHANGED <<currentTerm, votedFor, state, log, commitIndex, votesGranted,
                         votesRejected, electionTimeout, heartbeatTimeout, leader>>
       \/ /\ m.term >= currentTerm[n]
          /\ IF state[n] # StateFollower \/ currentTerm[n] # m.term
             THEN /\ currentTerm' = [currentTerm EXCEPT ![n] = m.term]
                  /\ state' = [state EXCEPT ![n] = StateFollower]
                  /\ votedFor' = [votedFor EXCEPT ![n] = CHOOSE x \in Nodes : TRUE]
                  /\ leader' = [leader EXCEPT ![n] = m.from]
             ELSE UNCHANGED <<currentTerm, votedFor, state, leader>>
          /\ commitIndex' = [commitIndex EXCEPT ![n] =
                             IF m.commitIndex > commitIndex[n] THEN m.commitIndex ELSE commitIndex[n]]
          /\ electionTimeout' = [electionTimeout EXCEPT ![n] = FALSE]
          /\ Discard(m)
          /\ UNCHANGED <<log, votesGranted, votesRejected, heartbeatTimeout>>

AdvanceCommitIndex(n) ==
    /\ state[n] = StateLeader
    /\ \E index \in (commitIndex[n] + 1)..Len(log[n]):
        /\ log[n][index] = currentTerm[n]
        /\ \E quorum \in Quorum:
            \A m \in quorum:
                \/ m = n
                \/ /\ Len(log[m]) >= index
                   /\ log[m][index] = log[n][index]
        /\ commitIndex' = [commitIndex EXCEPT ![n] = index]
    /\ UNCHANGED <<currentTerm, votedFor, state, log, votesGranted, votesRejected,
                   electionTimeout, heartbeatTimeout, leader, messages>>

ResetElectionTimeout(n) ==
    /\ electionTimeout[n] = FALSE
    /\ electionTimeout' = [electionTimeout EXCEPT ![n] = TRUE]
    /\ UNCHANGED <<currentTerm, votedFor, state, log, commitIndex, votesGranted,
                   votesRejected, heartbeatTimeout, leader, messages>>

ResetHeartbeatTimeout(n) ==
    /\ heartbeatTimeout[n] = FALSE
    /\ heartbeatTimeout' = [heartbeatTimeout EXCEPT ![n] = TRUE]
    /\ UNCHANGED <<currentTerm, votedFor, state, log, commitIndex, votesGranted,
                   votesRejected, electionTimeout, leader, messages>>

TypeOK ==
    /\ currentTerm \in [Nodes -> 0..MaxTerm]
    /\ votedFor \in [Nodes -> Nodes]
    /\ state \in [Nodes -> {StateFollower, StatePreCandidate, StateCandidate, StateLeader}]
    /\ log \in [Nodes -> Seq(0..MaxTerm)]
    /\ commitIndex \in [Nodes -> Nat]
    /\ votesGranted \in [Nodes -> SUBSET Nodes]
    /\ votesRejected \in [Nodes -> SUBSET Nodes]
    /\ electionTimeout \in [Nodes -> BOOLEAN]
    /\ heartbeatTimeout \in [Nodes -> BOOLEAN]
    /\ leader \in [Nodes -> Nodes]
    /\ messages \subseteq Message

Next ==
    \/ \E n \in Nodes: Timeout(n)
    \/ \E n \in Nodes: RequestPreVote(n)
    \/ \E n \in Nodes, m \in messages: HandlePreVoteRequest(n, m)
    \/ \E n \in Nodes, m \in messages: HandlePreVoteResponse(n, m)
    \/ \E n \in Nodes: RequestVote(n)
    \/ \E n \in Nodes, m \in messages: HandleVoteRequest(n, m)
    \/ \E n \in Nodes, m \in messages: HandleVoteResponse(n, m)
    \/ \E n \in Nodes: WinElection(n)
    \/ \E n \in Nodes, v \in 1..MaxTerm: ClientRequest(n, v)
    \/ \E n, m \in Nodes: SendAppendEntries(n, m)
    \/ \E n \in Nodes, m \in messages: HandleAppendEntriesRequest(n, m)
    \/ \E n \in Nodes, m \in messages: HandleAppendEntriesResponse(n, m)
    \/ \E n \in Nodes: SendHeartbeat(n)
    \/ \E n \in Nodes, m \in messages: HandleHeartbeat(n, m)
    \/ \E n \in Nodes: AdvanceCommitIndex(n)
    \/ \E n \in Nodes: ResetElectionTimeout(n)
    \/ \E n \in Nodes: ResetHeartbeatTimeout(n)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

====
