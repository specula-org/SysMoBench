MODULE etcdraft

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    Servers,
    MaxTerm,
    MaxLogLen,
    MaxCommittedEntries

VARIABLES
    currentTerm,
    votedFor,
    log,
    commitIndex,
    state,
    nextIndex,
    matchIndex,
    votes,
    leader,
    electionTimeout,
    heartbeatTimeout,
    messages,
    clientRequests,
    appliedIndex,
    keyValueStore

vars == <<currentTerm, votedFor, log, commitIndex, state, nextIndex, matchIndex, votes, leader, electionTimeout, heartbeatTimeout, messages, clientRequests, appliedIndex, keyValueStore>>

ServerStates == {"Follower", "Candidate", "Leader", "PreCandidate"}
MessageTypes == {"RequestVote", "RequestVoteResponse", "AppendEntries", "AppendEntriesResponse", "Heartbeat", "HeartbeatResponse", "ClientRequest", "ClientResponse"}

TypeOK ==
    /\ currentTerm \in [Servers -> 0..MaxTerm]
    /\ votedFor \in [Servers -> Servers \cup {Nil}]
    /\ log \in [Servers -> Seq(SUBSET (Nat \X Nat \X STRING))]
    /\ commitIndex \in [Servers -> Nat]
    /\ state \in [Servers -> ServerStates]
    /\ nextIndex \in [Servers -> [Servers -> Nat]]
    /\ matchIndex \in [Servers -> [Servers -> Nat]]
    /\ votes \in [Servers -> SUBSET Servers]
    /\ leader \in [Servers -> Servers \cup {Nil}]
    /\ electionTimeout \in [Servers -> Nat]
    /\ heartbeatTimeout \in [Servers -> Nat]
    /\ appliedIndex \in [Servers -> Nat]
    /\ keyValueStore \in [Servers -> [STRING -> STRING]]

Init ==
    /\ currentTerm = [s \in Servers |-> 0]
    /\ votedFor = [s \in Servers |-> Nil]
    /\ log = [s \in Servers |-> <<>>]
    /\ commitIndex = [s \in Servers |-> 0]
    /\ state = [s \in Servers |-> "Follower"]
    /\ nextIndex = [s \in Servers |-> [t \in Servers |-> 1]]
    /\ matchIndex = [s \in Servers |-> [t \in Servers |-> 0]]
    /\ votes = [s \in Servers |-> {}]
    /\ leader = [s \in Servers |-> Nil]
    /\ electionTimeout = [s \in Servers |-> 5]
    /\ heartbeatTimeout = [s \in Servers |-> 1]
    /\ messages = {}
    /\ clientRequests = {}
    /\ appliedIndex = [s \in Servers |-> 0]
    /\ keyValueStore = [s \in Servers |-> [k \in {} |-> ""]]

LastTerm(xlog) == IF Len(xlog) = 0 THEN 0 ELSE xlog[Len(xlog)].term
LastIndex(xlog) == Len(xlog)

IsUpToDate(i, j, xlog) ==
    \/ LastTerm(xlog) < i
    \/ /\ LastTerm(xlog) = i
       /\ LastIndex(xlog) <= j

Quorum == {S \in SUBSET Servers : Cardinality(S) * 2 > Cardinality(Servers)}

BecomeFollower(s, term) ==
    /\ currentTerm' = [currentTerm EXCEPT ![s] = term]
    /\ state' = [state EXCEPT ![s] = "Follower"]
    /\ votedFor' = [votedFor EXCEPT ![s] = Nil]
    /\ leader' = [leader EXCEPT ![s] = Nil]
    /\ votes' = [votes EXCEPT ![s] = {}]

BecomeCandidate(s) ==
    /\ state[s] \in {"Follower", "PreCandidate"}
    /\ currentTerm' = [currentTerm EXCEPT ![s] = currentTerm[s] + 1]
    /\ state' = [state EXCEPT ![s] = "Candidate"]
    /\ votedFor' = [votedFor EXCEPT ![s] = s]
    /\ votes' = [votes EXCEPT ![s] = {s}]
    /\ leader' = [leader EXCEPT ![s] = Nil]

BecomeLeader(s) ==
    /\ state[s] = "Candidate"
    /\ {s} \cup votes[s] \in Quorum
    /\ state' = [state EXCEPT ![s] = "Leader"]
    /\ leader' = [leader EXCEPT ![s] = s]
    /\ nextIndex' = [nextIndex EXCEPT ![s] = [t \in Servers |-> Len(log[s]) + 1]]
    /\ matchIndex' = [matchIndex EXCEPT ![s] = [t \in Servers |-> 0]]

RequestVote(s, t) ==
    /\ state[s] = "Candidate"
    /\ t \in Servers
    /\ t # s
    /\ messages' = messages \cup {[type |-> "RequestVote", 
                                   from |-> s, 
                                   to |-> t, 
                                   term |-> currentTerm[s],
                                   lastLogIndex |-> LastIndex(log[s]),
                                   lastLogTerm |-> LastTerm(log[s])]}

HandleRequestVote(s, m) ==
    /\ m.type = "RequestVote"
    /\ m.to = s
    /\ LET grant == /\ m.term >= currentTerm[s]
                    /\ \/ votedFor[s] = Nil
                       \/ votedFor[s] = m.from
                    /\ IsUpToDate(m.lastLogTerm, m.lastLogIndex, log[s])
       IN /\ IF m.term > currentTerm[s]
             THEN BecomeFollower(s, m.term)
             ELSE UNCHANGED <<currentTerm, state, votedFor, leader, votes>>
          /\ IF grant
             THEN /\ votedFor' = [votedFor EXCEPT ![s] = m.from]
                  /\ messages' = messages \cup {[type |-> "RequestVoteResponse",
                                                 from |-> s,
                                                 to |-> m.from,
                                                 term |-> currentTerm'[s],
                                                 voteGranted |-> TRUE]}
             ELSE /\ messages' = messages \cup {[type |-> "RequestVoteResponse",
                                                 from |-> s,
                                                 to |-> m.from,
                                                 term |-> currentTerm'[s],
                                                 voteGranted |-> FALSE]}
                  /\ UNCHANGED votedFor

HandleRequestVoteResponse(s, m) ==
    /\ m.type = "RequestVoteResponse"
    /\ m.to = s
    /\ state[s] = "Candidate"
    /\ IF m.term > currentTerm[s]
       THEN BecomeFollower(s, m.term)
       ELSE /\ IF m.voteGranted
               THEN votes' = [votes EXCEPT ![s] = votes[s] \cup {m.from}]
               ELSE UNCHANGED votes
            /\ UNCHANGED <<currentTerm, state, votedFor, leader>>

AppendEntries(s, t) ==
    /\ state[s] = "Leader"
    /\ t \in Servers
    /\ t # s
    /\ LET prevLogIndex == nextIndex[s][t] - 1
           prevLogTerm == IF prevLogIndex > 0 THEN log[s][prevLogIndex].term ELSE 0
           entries == SubSeq(log[s], nextIndex[s][t], Len(log[s]))
       IN messages' = messages \cup {[type |-> "AppendEntries",
                                       from |-> s,
                                       to |-> t,
                                       term |-> currentTerm[s],
                                       prevLogIndex |-> prevLogIndex,
                                       prevLogTerm |-> prevLogTerm,
                                       entries |-> entries,
                                       leaderCommit |-> commitIndex[s]]}

HandleAppendEntries(s, m) ==
    /\ m.type = "AppendEntries"
    /\ m.to = s
    /\ IF m.term > currentTerm[s]
       THEN BecomeFollower(s, m.term)
       ELSE UNCHANGED <<currentTerm, state, votedFor, leader, votes>>
    /\ LET logOk == \/ m.prevLogIndex = 0
                    \/ /\ m.prevLogIndex > 0
                       /\ m.prevLogIndex <= Len(log[s])
                       /\ log[s][m.prevLogIndex].term = m.prevLogTerm
       IN /\ IF logOk
             THEN /\ log' = [log EXCEPT ![s] = SubSeq(log[s], 1, m.prevLogIndex) \o m.entries]
                  /\ commitIndex' = [commitIndex EXCEPT ![s] = 
                                     IF m.leaderCommit > commitIndex[s]
                                     THEN Min(m.leaderCommit, Len(log'[s]))
                                     ELSE commitIndex[s]]
                  /\ messages' = messages \cup {[type |-> "AppendEntriesResponse",
                                                 from |-> s,
                                                 to |-> m.from,
                                                 term |-> currentTerm'[s],
                                                 success |-> TRUE,
                                                 matchIndex |-> Len(log'[s])]}
             ELSE /\ messages' = messages \cup {[type |-> "AppendEntriesResponse",
                                                 from |-> s,
                                                 to |-> m.from,
                                                 term |-> currentTerm'[s],
                                                 success |-> FALSE,
                                                 matchIndex |-> 0]}
                  /\ UNCHANGED <<log, commitIndex>>

HandleAppendEntriesResponse(s, m) ==
    /\ m.type = "AppendEntriesResponse"
    /\ m.to = s
    /\ state[s] = "Leader"
    /\ IF m.term > currentTerm[s]
       THEN BecomeFollower(s, m.term)
       ELSE /\ IF m.success
               THEN /\ nextIndex' = [nextIndex EXCEPT ![s][m.from] = m.matchIndex + 1]
                    /\ matchIndex' = [matchIndex EXCEPT ![s][m.from] = m.matchIndex]
               ELSE /\ nextIndex' = [nextIndex EXCEPT ![s][m.from] = Max(nextIndex[s][m.from] - 1, 1)]
                    /\ UNCHANGED matchIndex
            /\ UNCHANGED <<currentTerm, state, votedFor, leader, votes>>

ClientRequest(s, req) ==
    /\ state[s] = "Leader"
    /\ Len(log[s]) < MaxLogLen
    /\ LET entry == [term |-> currentTerm[s], 
                     index |-> Len(log[s]) + 1,
                     operation |-> req.operation,
                     key |-> req.key,
                     value |-> req.value]
       IN /\ log' = [log EXCEPT ![s] = Append(log[s], entry)]
          /\ clientRequests' = clientRequests \cup {req}

ApplyEntry(s) ==
    /\ appliedIndex[s] < commitIndex[s]
    /\ LET entry == log[s][appliedIndex[s] + 1]
       IN /\ appliedIndex' = [appliedIndex EXCEPT ![s] = appliedIndex[s] + 1]
          /\ IF entry.operation = "Put"
             THEN keyValueStore' = [keyValueStore EXCEPT ![s][entry.key] = entry.value]
             ELSE IF entry.operation = "Delete"
                  THEN keyValueStore' = [keyValueStore EXCEPT ![s] = 
                                         [k \in DOMAIN keyValueStore[s] \ {entry.key} |-> keyValueStore[s][k]]]
                  ELSE UNCHANGED keyValueStore

UpdateCommitIndex(s) ==
    /\ state[s] = "Leader"
    /\ LET newCommitIndex == CHOOSE i \in (commitIndex[s] + 1)..Len(log[s]) :
                                /\ log[s][i].term = currentTerm[s]
                                /\ Cardinality({t \in Servers : matchIndex[s][t] >= i}) * 2 > Cardinality(Servers)
       IN commitIndex' = [commitIndex EXCEPT ![s] = newCommitIndex]

Timeout(s) ==
    /\ state[s] \in {"Follower", "Candidate"}
    /\ electionTimeout[s] = 0
    /\ BecomeCandidate(s)
    /\ electionTimeout' = [electionTimeout EXCEPT ![s] = 5]

Heartbeat(s) ==
    /\ state[s] = "Leader"
    /\ heartbeatTimeout[s] = 0
    /\ \A t \in Servers \ {s} : AppendEntries(s, t)
    /\ heartbeatTimeout' = [heartbeatTimeout EXCEPT ![s] = 1]

TickElectionTimeout(s) ==
    /\ electionTimeout[s] > 0
    /\ electionTimeout' = [electionTimeout EXCEPT ![s] = electionTimeout[s] - 1]

TickHeartbeatTimeout(s) ==
    /\ heartbeatTimeout[s] > 0
    /\ heartbeatTimeout' = [heartbeatTimeout EXCEPT ![s] = heartbeatTimeout[s] - 1]

ReceiveMessage(m) ==
    /\ m \in messages
    /\ \/ HandleRequestVote(m.to, m)
       \/ HandleRequestVoteResponse(m.to, m)
       \/ HandleAppendEntries(m.to, m)
       \/ HandleAppendEntriesResponse(m.to, m)
    /\ messages' = messages \ {m}

Next ==
    \/ \E s \in Servers : 
        \/ Timeout(s)
        \/ TickElectionTimeout(s)
        \/ TickHeartbeatTimeout(s)
        \/ Heartbeat(s)
        \/ UpdateCommitIndex(s)
        \/ ApplyEntry(s)
        \/ \E t \in Servers \ {s} : RequestVote(s, t)
        \/ \E t \in Servers \ {s} : AppendEntries(s, t)
        \/ \E req \in clientRequests : ClientRequest(s, req)
    \/ \E m \in messages : ReceiveMessage(m)

Spec == Init /\ [][Next]_vars

ElectionSafety == 
    \A s, t \in Servers : 
        /\ state[s] = "Leader" 
        /\ state[t] = "Leader" 
        /\ currentTerm[s] = currentTerm[t]
        => s = t

LeaderAppendOnly ==
    \A s \in Servers :
        state[s] = "Leader" =>
            \A i \in 1..Len(log[s]) :
                log[s][i].term <= currentTerm[s]

LogMatching ==
    \A s, t \in Servers :
        \A i \in 1..Min(Len(log[s]), Len(log[t])) :
            log[s][i].term = log[t][i].term =>
                \A j \in 1..i : log[s][j] = log[t][j]

LeaderCompleteness ==
    \A s \in Servers :
        state[s]= "Leader" => 
            commitIndex[s] = Len(log[s]) 

Linearizability ==
    \A req \in clientRequests :
        \A s \in Servers :
            req.operation = "Get" => 
                \A i \in 1..Len(log[s]) :
                    log[s][i].operation = "Put" /\ log[s][i].key = req.key => 
                        log[s][i].value = keyValueStore[s][req.key]

Theorem Spec => ElectionSafety
Theorem Spec => LeaderAppendOnly
Theorem Spec => LogMatching
Theorem Spec => LeaderCompleteness
Theorem Spec => Linearizability