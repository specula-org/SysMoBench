---- MODULE etcdraft ----
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS Servers,         \* The set of server IDs
          Commands,        \* Abstract client commands (keys and values)
          Nil,             \* A value distinct from any server ID
          HeartbeatTimeout,\* Ticks for leader to send heartbeats
          ElectionTimeout  \* Base ticks for election timeout

ASSUME /\ TypeOK
       /\ HeartbeatTimeout > 0
       /\ ElectionTimeout > HeartbeatTimeout

\* -- Model Values --
ServerStates == {"Follower", "Candidate", "PreCandidate", "Leader"}
MsgTypes == {"MsgApp", "MsgAppResp", "MsgVote", "MsgVoteResp",
             "MsgPreVote", "MsgPreVoteResp", "MsgTransferLeader", "MsgTimeoutNow"}

\* -- Configuration --
PreVoteEnabled == TRUE

\* -- State Variables --
vars == <<
    state, currentTerm, votedFor, log,
    commitIndex, appliedIndex,
    leader, leadTransferee,
    nextIndex, matchIndex,
    electionElapsed, heartbeatElapsed, randomizedElectionTimeout,
    messages,
    kvStore
>>

\* -- Helper Functions --
QuorumSize == (Cardinality(Servers) \div 2) + 1

LastLogIndex(s) == Len(log[s])
LastLogTerm(s) == IF LastLogIndex(s) = 0 THEN 0 ELSE log[s][LastLogIndex(s)].term

\* Log up-to-dateness check from Raft paper section 5.4.1
IsUpToDate(candidateTerm, candidateIndex, voterLog) ==
    LET voterLastTerm == IF Len(voterLog) = 0 THEN 0 ELSE voterLog[Len(voterLog)].term
        voterLastIndex == Len(voterLog)
    IN (candidateTerm > voterLastTerm) \/
       ((candidateTerm = voterLastTerm) /\ (candidateIndex >= voterLastIndex))

NewRandomizedTimeout(s) ==
    CHOOSE t \in ElectionTimeout..(2*ElectionTimeout - 1) : TRUE

\* -- State Transitions --

BecomeFollower(s, term, l) ==
    /\ state' = [state EXCEPT ![s] = "Follower"]
    /\ currentTerm' = [currentTerm EXCEPT ![s] = term]
    /\ votedFor' = [votedFor EXCEPT ![s] = Nil]
    /\ leader' = [leader EXCEPT ![s] = l]
    /\ electionElapsed' = [electionElapsed EXCEPT ![s] = 0]
    /\ randomizedElectionTimeout' = [randomizedElectionTimeout EXCEPT ![s] = NewRandomizedTimeout(s)]
    /\ leadTransferee' = [leadTransferee EXCEPT ![s] = Nil]

BecomeCandidate(s) ==
    /\ state' = [state EXCEPT ![s] = "Candidate"]
    /\ currentTerm' = [currentTerm EXCEPT ![s] = currentTerm[s] + 1]
    /\ votedFor' = [votedFor EXCEPT ![s] = s]
    /\ leader' = [leader EXCEPT ![s] = Nil]
    /\ electionElapsed' = [electionElapsed EXCEPT ![s] = 0]
    /\ randomizedElectionTimeout' = [randomizedElectionTimeout EXCEPT ![s] = NewRandomizedTimeout(s)]
    /\ messages' = messages \cup
        { [ type |-> "MsgVote", from |-> s, to |-> t, term |-> currentTerm[s] + 1,
            logTerm |-> LastLogTerm(s), index |-> LastLogIndex(s) ]
          : t \in Servers \ {s} }

BecomePreCandidate(s) ==
    /\ state' = [state EXCEPT ![s] = "PreCandidate"]
    /\ leader' = [leader EXCEPT ![s] = Nil]
    /\ electionElapsed' = [electionElapsed EXCEPT ![s] = 0]
    /\ randomizedElectionTimeout' = [randomizedElectionTimeout EXCEPT ![s] = NewRandomizedTimeout(s)]
    /\ messages' = messages \cup
        { [ type |-> "MsgPreVote", from |-> s, to |-> t, term |-> currentTerm[s] + 1,
            logTerm |-> LastLogTerm(s), index |-> LastLogIndex(s) ]
          : t \in Servers \ {s} }

BecomeLeader(s) ==
    /\ state' = [state EXCEPT ![s] = "Leader"]
    /\ leader' = [leader EXCEPT ![s] = s]
    /\ leadTransferee' = [leadTransferee EXCEPT ![s] = Nil]
    /\ nextIndex' = [nextIndex EXCEPT ![s] = [t \in Servers |-> LastLogIndex(s) + 1]]
    /\ matchIndex' = [matchIndex EXCEPT ![s] = [t \in Servers |-> 0] @@ [s |-> LastLogIndex(s) + 1]]
    /\ heartbeatElapsed' = [heartbeatElapsed EXCEPT ![s] = 0]
    /\ log' = [log EXCEPT ![s] = Append(log[s], [term |-> currentTerm[s], command |-> "NoOp"])]
    /\ messages' = messages \cup
        { [ type |-> "MsgApp", from |-> s, to |-> t, term |-> currentTerm[s],
            prevLogIndex |-> LastLogIndex(s),
            prevLogTerm |-> LastLogTerm(s),
            entries |-> << [term |-> currentTerm[s], command |-> "NoOp"] >>,
            commit |-> commitIndex[s] ]
          : t \in Servers \ {s} }

\* A follower, candidate or pre-candidate times out and starts an election.
Timeout(s) ==
    /\ state[s] \in {"Follower", "Candidate", "PreCandidate"}
    /\ electionElapsed[s] >= randomizedElectionTimeout[s]
    /\ IF PreVoteEnabled
       THEN BecomePreCandidate(s)
       ELSE BecomeCandidate(s)
    /\ UNCHANGED <<commitIndex, appliedIndex, log, nextIndex, matchIndex, heartbeatElapsed, leadTransferee, kvStore>>

\* A leader's heartbeat timer expires, so it sends heartbeats to its peers.
LeaderHeartbeat(s) ==
    /\ state[s] = "Leader"
    /\ heartbeatElapsed[s] >= HeartbeatTimeout
    /\ heartbeatElapsed' = [heartbeatElapsed EXCEPT ![s] = 0]
    /\ messages' = messages \cup
        { [ type |-> "MsgApp", from |-> s, to |-> t, term |-> currentTerm[s],
            entries |-> << >>, commit |-> min(commitIndex[s], matchIndex[s][t]) ]
          : t \in Servers \ {s} }
    /\ UNCHANGED <<state, currentTerm, votedFor, log, commitIndex, appliedIndex, leader, leadTransferee, nextIndex, matchIndex, electionElapsed, randomizedElectionTimeout, kvStore>>

\* A client sends a command to be replicated.
ClientRequest(s, cmd) ==
    /\ state[s] = "Leader"
    /\ leadTransferee[s] = Nil
    /\ LET newEntry == [term |-> currentTerm[s], command |-> cmd]
           newLog == Append(log[s], newEntry)
    IN
    /\ log' = [log EXCEPT ![s] = newLog]
    /\ matchIndex' = [matchIndex EXCEPT ![s][s] = LastLogIndex(newLog)]
    /\ messages' = messages \cup
        { [ type |-> "MsgApp", from |-> s, to |-> t, term |-> currentTerm[s],
            prevLogIndex |-> LastLogIndex(s),
            prevLogTerm |-> LastLogTerm(s),
            entries |-> << newEntry >>,
            commit |-> commitIndex[s] ]
          : t \in Servers \ {s} }
    /\ UNCHANGED <<state, currentTerm, votedFor, commitIndex, appliedIndex, leader, leadTransferee, nextIndex, electionElapsed, heartbeatElapsed, randomizedElectionTimeout, kvStore>>

\* A server s handles an AppendEntries RPC.
HandleAppendEntries(s, m) ==
    /\ m.type = "MsgApp"
    /\ m.to = s
    /\ m \in messages
    /\ \/ /\ m.term < currentTerm[s]
          /\ messages' = (messages \ {m}) \cup
                {[ type |-> "MsgAppResp", from |-> s, to |-> m.from,
                   term |-> currentTerm[s], reject |-> TRUE, index |-> 0 ]}
          /\ UNCHANGED <<vars \ {messages}>>
       \/ /\ m.term >= currentTerm[s]
          /\ BecomeFollower(s, m.term, m.from)
          /\ LET
              logOK == /\ m.prevLogIndex = 0
                       \/ /\ m.prevLogIndex > 0
                          /\ m.prevLogIndex <= LastLogIndex(s)
                          /\ log[s][m.prevLogIndex].term = m.prevLogTerm
          IN
          /\ IF logOK
             THEN /\ LET
                      \* Find insertion point and truncate if there's a conflict
                      conflictIndex == CHOOSE i \in 1..Len(m.entries) :
                                         (m.prevLogIndex + i > LastLogIndex(s)) \/
                                         (log[s][m.prevLogIndex + i].term /= m.entries[i].term)
                      newEntries == SubSeq(m.entries, conflictIndex, Len(m.entries))
                      prefix == SubSeq(log[s], 1, m.prevLogIndex + conflictIndex - 1)
                      newLog == prefix \o newEntries
                  IN
                  /\ log' = [log EXCEPT ![s] = newLog]
                  /\ IF m.commit > commitIndex[s]
                     THEN commitIndex' = [commitIndex EXCEPT ![s] = min(m.commit, LastLogIndex(newLog))]
                     ELSE UNCHANGED commitIndex
                  /\ messages' = (messages \ {m}) \cup
                      {[ type |-> "MsgAppResp", from |-> s, to |-> m.from,
                         term |-> currentTerm[s], reject |-> FALSE,
                         index |-> LastLogIndex(newLog) ]}
             ELSE /\ messages' = (messages \ {m}) \cup
                      {[ type |-> "MsgAppResp", from |-> s, to |-> m.from,
                         term |-> currentTerm[s], reject |-> TRUE,
                         index |-> m.prevLogIndex, rejectHint |-> commitIndex[s] ]}
                  /\ UNCHANGED <<log, commitIndex>>
          /\ UNCHANGED <<votedFor, appliedIndex, nextIndex, matchIndex, heartbeatElapsed, kvStore>>

\* A leader s handles an AppendEntries response.
HandleAppendEntriesResponse(s, m) ==
    /\ m.type = "MsgAppResp"
    /\ m.to = s
    /\ m \in messages
    /\ state[s] = "Leader"
    /\ m.term = currentTerm[s]
    /\ LET follower == m.from
    IN
    /\ IF m.reject = FALSE
       THEN /\ nextIndex' = [nextIndex EXCEPT ![s][follower] = m.index + 1]
            /\ matchIndex' = [matchIndex EXCEPT ![s][follower] = m.index]
            /\ IF leadTransferee[s] = follower /\ m.index = LastLogIndex(s)
               THEN messages' = (messages \ {m}) \cup
                                 {[ type |-> "MsgTimeoutNow", from |-> s, to |-> follower, term |-> currentTerm[s] ]}
               ELSE messages' = messages \ {m}
       ELSE /\ nextIndex' = [nextIndex EXCEPT ![s][follower] = max(1, m.rejectHint + 1)]
            /\ messages' = messages \ {m}
            /\ UNCHANGED matchIndex
    /\ UNCHANGED <<state, currentTerm, votedFor, log, commitIndex, appliedIndex, leader, leadTransferee, electionElapsed, heartbeatElapsed, randomizedElectionTimeout, kvStore>>

\* A server s handles a RequestVote RPC.
HandleRequestVote(s, m) ==
    /\ m.type = "MsgVote"
    /\ m.to = s
    /\ m \in messages
    /\ LET
        grant == /\ m.term >= currentTerm[s]
                 /\ (votedFor[s] = Nil \/ votedFor[s] = m.from)
                 /\ IsUpToDate(m.logTerm, m.index, log[s])
    IN
    /\ IF grant
       THEN /\ currentTerm' = [currentTerm EXCEPT ![s] = m.term]
            /\ votedFor' = [votedFor EXCEPT ![s] = m.from]
            /\ state' = [state EXCEPT ![s] = "Follower"]
            /\ messages' = (messages \ {m}) \cup
                {[ type |-> "MsgVoteResp", from |-> s, to |-> m.from,
                   term |-> m.term, reject |-> FALSE ]}
       ELSE /\ messages' = (messages \ {m}) \cup
                {[ type |-> "MsgVoteResp", from |-> s, to |-> m.from,
                   term |-> currentTerm[s], reject |-> TRUE ]}
            /\ UNCHANGED <<currentTerm, votedFor, state>>
    /\ UNCHANGED <<log, commitIndex, appliedIndex, leader, leadTransferee, nextIndex, matchIndex, electionElapsed, heartbeatElapsed, randomizedElectionTimeout, kvStore>>

\* A candidate s handles a RequestVote response.
HandleRequestVoteResponse(s, m) ==
    /\ m.type = "MsgVoteResp"
    /\ m.to = s
    /\ m \in messages
    /\ state[s] = "Candidate"
    /\ \/ /\ m.term > currentTerm[s]
          /\ BecomeFollower(s, m.term, Nil)
          /\ messages' = messages \ {m}
          /\ UNCHANGED <<log, commitIndex, appliedIndex, nextIndex, matchIndex, heartbeatElapsed, kvStore>>
       \/ /\ m.term = currentTerm[s]
          /\ LET
              votesGranted == {t \in Servers | m.from = t /\ m.reject = FALSE}
              newVotes == votes[s] \cup votesGranted
              quorumMet == Cardinality(newVotes) >= QuorumSize
          IN
          /\ IF quorumMet
             THEN BecomeLeader(s)
             ELSE UNCHANGED <<state, currentTerm, leader, leadTransferee, nextIndex, matchIndex, heartbeatElapsed, log>>
          /\ votes' = [votes EXCEPT ![s] = newVotes]
          /\ messages' = messages \ {m}
          /\ UNCHANGED <<votedFor, commitIndex, appliedIndex, electionElapsed, randomizedElectionTimeout, kvStore>>

\* A server s handles a PreVote RPC.
HandlePreVote(s, m) ==
    /\ m.type = "MsgPreVote"
    /\ m.to = s
    /\ m \in messages
    /\ LET
        grant == /\ m.term > currentTerm[s]
                 /\ IsUpToDate(m.logTerm, m.index, log[s])
    IN
    /\ messages' = (messages \ {m}) \cup
        {[ type |-> "MsgPreVoteResp", from |-> s, to |-> m.from,
           term |-> m.term, reject |-> ~grant ]}
    /\ UNCHANGED <<vars \ {messages}>>

\* A pre-candidate s handles a PreVote response.
HandlePreVoteResponse(s, m) ==
    /\ m.type = "MsgPreVoteResp"
    /\ m.to = s
    /\ m \in messages
    /\ state[s] = "PreCandidate"
    /\ m.term = currentTerm[s] + 1
    /\ LET
        preVotesGranted == {t \in Servers | m.from = t /\ m.reject = FALSE}
        newPreVotes == preVotes[s] \cup preVotesGranted
        quorumMet == Cardinality(newPreVotes) >= QuorumSize
    IN
    /\ IF quorumMet
       THEN BecomeCandidate(s)
       ELSE UNCHANGED <<state, currentTerm, votedFor, leader, leadTransferee, nextIndex, matchIndex, heartbeatElapsed, log, messages>>
    /\ preVotes' = [preVotes EXCEPT ![s] = newPreVotes]
    /\ messages' = messages \ {m}
    /\ UNCHANGED <<commitIndex, appliedIndex, electionElapsed, randomizedElectionTimeout, kvStore>>

\* A leader s advances its commit index.
LeaderAdvanceCommit(s) ==
    /\ state[s] = "Leader"
    /\ LET
        PossibleCommits ==
            { i \in (commitIndex[s]+1)..LastLogIndex(s) :
                /\ log[s][i].term = currentTerm[s]
                /\ Cardinality({t \in Servers : matchIndex[s][t] >= i}) >= QuorumSize
            }
    IN
    /\ IF PossibleCommits /= {}
       THEN commitIndex' = [commitIndex EXCEPT ![s] = Max(PossibleCommits)]
       ELSE UNCHANGED commitIndex
    /\ UNCHANGED <<state, currentTerm, votedFor, log, appliedIndex, leader, leadTransferee, nextIndex, matchIndex, electionElapsed, heartbeatElapsed, randomizedElectionTimeout, messages, kvStore>>

\* A server s applies a committed entry to its state machine.
Apply(s) ==
    /\ commitIndex[s] > appliedIndex[s]
    /\ LET i == appliedIndex[s] + 1
           entry == log[s][i]
           cmd == entry.command
    IN
    /\ appliedIndex' = [appliedIndex EXCEPT ![s] = i]
    /\ IF cmd /= "NoOp"
       THEN kvStore' = [kvStore EXCEPT ![s][cmd.key] = cmd.val]
       ELSE UNCHANGED kvStore
    /\ UNCHANGED <<state, currentTerm, votedFor, log, commitIndex, leader, leadTransferee, nextIndex, matchIndex, electionElapsed, heartbeatElapsed, randomizedElectionTimeout, messages>>

\* A leader s handles a request to transfer leadership.
HandleTransferLeader(s, m) ==
    /\ m.type = "MsgTransferLeader"
    /\ m.to = s
    /\ m \in messages
    /\ state[s] = "Leader"
    /\ LET transferee == m.from
    IN
    /\ leadTransferee' = [leadTransferee EXCEPT ![s] = transferee]
    /\ IF matchIndex[s][transferee] = LastLogIndex(s)
       THEN messages' = (messages \ {m}) \cup
                         {[ type |-> "MsgTimeoutNow", from |-> s, to |-> transferee, term |-> currentTerm[s] ]}
       ELSE messages' = messages \ {m}
    /\ UNCHANGED <<state, currentTerm, votedFor, log, commitIndex, appliedIndex, leader, nextIndex, matchIndex, electionElapsed, heartbeatElapsed, randomizedElectionTimeout, kvStore>>

\* A server s is told to time out immediately to start an election.
HandleTimeoutNow(s, m) ==
    /\ m.type = "MsgTimeoutNow"
    /\ m.to = s
    /\ m \in messages
    /\ m.term = currentTerm[s]
    /\ IF PreVoteEnabled
       THEN BecomePreCandidate(s)
       ELSE BecomeCandidate(s)
    /\ messages' = messages \ {m}
    /\ UNCHANGED <<commitIndex, appliedIndex, log, nextIndex, matchIndex, heartbeatElapsed, leadTransferee, kvStore>>

\* A message is dropped from the network.
DropMessage(m) ==
    /\ m \in messages
    /\ messages' = messages \ {m}
    /\ UNCHANGED <<state, currentTerm, votedFor, log, commitIndex, appliedIndex, leader, leadTransferee, nextIndex, matchIndex, electionElapsed, heartbeatElapsed, randomizedElectionTimeout, kvStore>>

\* A tick of a logical clock passes for a server.
Tick(s) ==
    /\ electionElapsed' = [electionElapsed EXCEPT ![s] = electionElapsed[s] + 1]
    /\ IF state[s] = "Leader"
       THEN heartbeatElapsed' = [heartbeatElapsed EXCEPT ![s] = heartbeatElapsed[s] + 1]
       ELSE UNCHANGED heartbeatElapsed
    /\ UNCHANGED <<state, currentTerm, votedFor, log, commitIndex, appliedIndex, leader, leadTransferee, nextIndex, matchIndex, randomizedElectionTimeout, messages, kvStore>>

\* -- Initial State --
Init ==
    /\ state = [s \in Servers |-> "Follower"]
    /\ currentTerm = [s \in Servers |-> 0]
    /\ votedFor = [s \in Servers |-> Nil]
    /\ log = [s \in Servers |-> << >>]
    /\ commitIndex = [s \in Servers |-> 0]
    /\ appliedIndex = [s \in Servers |-> 0]
    /\ leader = [s \in Servers |-> Nil]
    /\ leadTransferee = [s \in Servers |-> Nil]
    /\ nextIndex = [s \in Servers |-> [t \in Servers |-> 1]]
    /\ matchIndex = [s \in Servers |-> [t \in Servers |-> 0]]
    /\ electionElapsed = [s \in Servers |-> 0]
    /\ heartbeatElapsed = [s \in Servers |-> 0]
    /\ randomizedElectionTimeout = [s \in Servers |-> NewRandomizedTimeout(s)]
    /\ messages = {}
    /\ kvStore = [s \in Servers |-> [k \in {} |-> ""]]
    /\ LET V(s) == {} IN votes = [s \in Servers |-> V(s)]
    /\ LET PV(s) == {} IN preVotes = [s \in Servers |-> PV(s)]

\* -- Next-State Relation --
Next ==
    \/ \E s \in Servers: Timeout(s)
    \/ \E s \in Servers: LeaderHeartbeat(s)
    \/ \E s \in Servers, cmd \in Commands: ClientRequest(s, cmd)
    \/ \E s \in Servers, m \in messages:
        \/ HandleAppendEntries(s, m)
        \/ HandleAppendEntriesResponse(s, m)
        \/ HandleRequestVote(s, m)
        \/ HandleRequestVoteResponse(s, m)
        \/ HandlePreVote(s, m)
        \/ HandlePreVoteResponse(s, m)
        \/ HandleTransferLeader(s, m)
        \/ HandleTimeoutNow(s, m)
    \/ \E s \in Servers: LeaderAdvanceCommit(s)
    \/ \E s \in Servers: Apply(s)
    \/ \E m \in messages: DropMessage(m)
    \/ \E s \in Servers: Tick(s)

\* -- Specification --
Spec == Init /\ [][Next]_vars

\* -- Invariants --
TypeOK ==
    /\ state \in [Servers -> ServerStates]
    /\ currentTerm \in [Servers -> Nat]
    /\ votedFor \in [Servers -> Servers \cup {Nil}]
    /\ \A s \in Servers: \A e \in log[s]: e \in [term: Nat, command: Commands \cup {"NoOp"}]
    /\ commitIndex \in [Servers -> Nat]
    /\ appliedIndex \in [Servers -> Nat]
    /\ leader \in [Servers -> Servers \cup {Nil}]
    /\ leadTransferee \in [Servers -> Servers \cup {Nil}]
    /\ messages \subseteq
        [ type: MsgTypes, from: Servers, to: Servers, term: Nat,
          logTerm: Nat, index: Nat, prevLogIndex: Nat, prevLogTerm: Nat,
          entries: Seq([term: Nat, command: Commands \cup {"NoOp"}]),
          commit: Nat, reject: BOOLEAN, rejectHint: Nat ]

\* At most one leader per term.
LeaderElectionSafety ==
    \A t \in {currentTerm[s] : s \in Servers}:
        Cardinality({s \in Servers : state[s] = "Leader" /\ currentTerm[s] = t}) <= 1

\* A leader's log is never overwritten.
LeaderLogsOnlyAppend ==
    \A s \in Servers:
        state[s] = "Leader" => \A e \in log[s]: e \in log'[s]

\* If two logs contain an entry with the same index and term,
\* then the logs are identical up to that index.
LogMatchingProperty ==
    \A s1, s2 \in Servers:
        \A i \in 1..min(LastLogIndex(s1), LastLogIndex(s2)):
            (log[s1][i].term = log[s2][i].term) => (log[s1][i] = log[s2][i])

\* All committed entries are present in the logs of subsequent leaders.
LeaderCompleteness ==
    \A i \in 1..LastLogIndex(s1):
        (log[s1][i].term < currentTerm[s2] /\ state[s2] = "Leader") =>
            (i <= LastLogIndex(s2) /\ log[s2][i] = log[s1][i])

\* If a server has applied an entry, that entry must be committed.
StateMaschineSafety ==
    \A s \in Servers: appliedIndex[s] <= commitIndex[s]

\* If two servers have applied entries up to the same index, their
\* state machines must be identical.
Linearizability ==
    \A s1, s2 \in Servers:
        (appliedIndex[s1] = appliedIndex[s2]) => (kvStore[s1] = kvStore[s2])

=============================================================================