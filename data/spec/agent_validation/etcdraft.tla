---- MODULE etcdraft ----
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS 
    Servers,
    MaxTerm,
    MaxLogLen,
    MaxCommittedSizePerReady,
    MaxInflightMsgs,
    ElectionTimeout,
    HeartbeatTimeout

VARIABLES
    currentTerm,
    votedFor,
    log,
    commitIndex,
    state,
    leader,
    nextIndex,
    matchIndex,
    votes,
    electionElapsed,
    heartbeatElapsed,
    randomizedElectionTimeout,
    messages,
    uncommittedSize,
    appliedIndex,
    keyValueStore,
    clientRequests,
    clientResponses,
    readStates,
    pendingReadIndex,
    leadTransferee,
    isLearner,
    recentActive,
    inflightMsgs,
    pendingConfIndex

vars == <<currentTerm, votedFor, log, commitIndex, state, leader, nextIndex, matchIndex, votes, 
          electionElapsed, heartbeatElapsed, randomizedElectionTimeout, messages, uncommittedSize,
          appliedIndex, keyValueStore, clientRequests, clientResponses, readStates, pendingReadIndex,
          leadTransferee, isLearner, recentActive, inflightMsgs, pendingConfIndex>>

States == {"Follower", "Candidate", "Leader", "PreCandidate"}
MessageTypes == {"RequestVote", "RequestVoteResponse", "AppendEntries", "AppendEntriesResponse", 
                 "Heartbeat", "HeartbeatResponse", "ClientRequest", "ClientResponse", "ReadIndex",
                 "ReadIndexResponse", "Snapshot", "SnapshotResponse", "TimeoutNow", "TransferLeader"}
EntryTypes == {"Normal", "ConfChange", "ConfChangeV2"}
ClientRequestTypes == {"Get", "Put", "Delete"}
ClientResponseTypes == {"GetResponse", "PutResponse", "DeleteResponse"}

None == 0
Nil == 0

Min(a, b) == IF a < b THEN a ELSE b
Max(a, b) == IF a > b THEN a ELSE b

TypeOK ==
    /\ currentTerm \in [Servers -> 0..MaxTerm]
    /\ votedFor \in [Servers -> Servers \cup {None}]
    /\ log \in [Servers -> Seq([term: 0..MaxTerm, type: EntryTypes, data: STRING])]
    /\ commitIndex \in [Servers -> Nat]
    /\ state \in [Servers -> States]
    /\ leader \in [Servers -> Servers \cup {None}]
    /\ nextIndex \in [Servers -> [Servers -> Nat]]
    /\ matchIndex \in [Servers -> [Servers -> Nat]]
    /\ votes \in [Servers -> SUBSET Servers]
    /\ electionElapsed \in [Servers -> Nat]
    /\ heartbeatElapsed \in [Servers -> Nat]
    /\ randomizedElectionTimeout \in [Servers -> Nat]
    /\ messages \subseteq [type: MessageTypes, from: Servers, to: Servers, term: 0..MaxTerm,
                          logTerm: 0..MaxTerm, index: Nat, entries: Seq(STRING),
                          commit: Nat, success: BOOLEAN, data: STRING, context: STRING]
    /\ uncommittedSize \in [Servers -> Nat]
    /\ appliedIndex \in [Servers -> Nat]
    /\ keyValueStore \in [Servers -> [STRING -> STRING]]
    /\ clientRequests \subseteq [type: ClientRequestTypes, key: STRING, value: STRING, client: STRING]
    /\ clientResponses \subseteq [type: ClientResponseTypes, 
                                 key: STRING, value: STRING, client: STRING, success: BOOLEAN]
    /\ readStates \in [Servers -> Seq([index: Nat, requestCtx: STRING])]
    /\ pendingReadIndex \in [Servers -> Seq([type: MessageTypes, from: Servers, to: Servers, 
                                           term: 0..MaxTerm, data: STRING])]
    /\ leadTransferee \in [Servers -> Servers \cup {None}]
    /\ isLearner \in [Servers -> BOOLEAN]
    /\ recentActive \in [Servers -> [Servers -> BOOLEAN]]
    /\ inflightMsgs \in [Servers -> [Servers -> Nat]]
    /\ pendingConfIndex \in [Servers -> Nat]

Init ==
    /\ currentTerm = [s \in Servers |-> 0]
    /\ votedFor = [s \in Servers |-> None]
    /\ log = [s \in Servers |-> <<>>]
    /\ commitIndex = [s \in Servers |-> 0]
    /\ state = [s \in Servers |-> "Follower"]
    /\ leader = [s \in Servers |-> None]
    /\ nextIndex = [s \in Servers |-> [t \in Servers |-> 1]]
    /\ matchIndex = [s \in Servers |-> [t \in Servers |-> 0]]
    /\ votes = [s \in Servers |-> {}]
    /\ electionElapsed = [s \in Servers |-> 0]
    /\ heartbeatElapsed = [s \in Servers |-> 0]
    /\ randomizedElectionTimeout = [s \in Servers |-> ElectionTimeout + (s % ElectionTimeout)]
    /\ messages = {}
    /\ uncommittedSize = [s \in Servers |-> 0]
    /\ appliedIndex = [s \in Servers |-> 0]
    /\ keyValueStore = [s \in Servers |-> [k \in {} |-> ""]]
    /\ clientRequests = {}
    /\ clientResponses = {}
    /\ readStates = [s \in Servers |-> <<>>]
    /\ pendingReadIndex = [s \in Servers |-> <<>>]
    /\ leadTransferee = [s \in Servers |-> None]
    /\ isLearner = [s \in Servers |-> FALSE]
    /\ recentActive = [s \in Servers |-> [t \in Servers |-> TRUE]]
    /\ inflightMsgs = [s \in Servers |-> [t \in Servers |-> 0]]
    /\ pendingConfIndex = [s \in Servers |-> 0]

LastTerm(xlog) == IF Len(xlog) = 0 THEN 0 ELSE xlog[Len(xlog)].term
LastIndex(xlog) == Len(xlog)
LogTerm(xlog, i) == IF i = 0 \/ i > Len(xlog) THEN 0 ELSE xlog[i].term

IsUpToDate(i, term, xlog) ==
    \/ LastTerm(xlog) < term
    \/ /\ LastTerm(xlog) = term
       /\ LastIndex(xlog) <= i

CanVote(s, candidate, candidateTerm, candidateIndex, candidateLogTerm) ==
    /\ \/ votedFor[s] = None
       \/ votedFor[s] = candidate
    /\ IsUpToDate(candidateIndex, candidateLogTerm, log[s])

Quorum == {S \in SUBSET Servers : Cardinality(S) * 2 > Cardinality(Servers)}

BecomeFollower(s, term, newLeader) ==
    /\ state' = [state EXCEPT ![s] = "Follower"]
    /\ currentTerm' = [currentTerm EXCEPT ![s] = term]
    /\ leader' = [leader EXCEPT ![s] = newLeader]
    /\ votedFor' = [votedFor EXCEPT ![s] = None]
    /\ electionElapsed' = [electionElapsed EXCEPT ![s] = 0]
    /\ heartbeatElapsed' = [heartbeatElapsed EXCEPT ![s] = 0]
    /\ votes' = [votes EXCEPT ![s] = {}]
    /\ leadTransferee' = [leadTransferee EXCEPT ![s] = None]

BecomeCandidate(s) ==
    /\ state' = [state EXCEPT ![s] = "Candidate"]
    /\ currentTerm' = [currentTerm EXCEPT ![s] = currentTerm[s] + 1]
    /\ votedFor' = [votedFor EXCEPT ![s] = s]
    /\ leader' = [leader EXCEPT ![s] = None]
    /\ electionElapsed' = [electionElapsed EXCEPT ![s] = 0]
    /\ votes' = [votes EXCEPT ![s] = {s}]

BecomeLeader(s) ==
    /\ state' = [state EXCEPT ![s] = "Leader"]
    /\ leader' = [leader EXCEPT ![s] = s]
    /\ nextIndex' = [nextIndex EXCEPT ![s] = [t \in Servers |-> LastIndex(log[s]) + 1]]
    /\ matchIndex' = [matchIndex EXCEPT ![s] = [t \in Servers |-> 0]]
    /\ heartbeatElapsed' = [heartbeatElapsed EXCEPT ![s] = 0]
    /\ pendingConfIndex' = [pendingConfIndex EXCEPT ![s] = LastIndex(log[s])]

Timeout(s) ==
    /\ state[s] \in {"Follower", "Candidate"}
    /\ electionElapsed[s] >= randomizedElectionTimeout[s]
    /\ BecomeCandidate(s)
    /\ LET requestVoteMessages == 
           {[type |-> "RequestVote", from |-> s, to |-> t, term |-> currentTerm'[s],
             logTerm |-> LastTerm(log[s]), index |-> LastIndex(log[s]),
             entries |-> <<>>, commit |-> 0, success |-> FALSE, data |-> "", context |-> ""] 
            : t \in Servers \ {s}}
       IN messages' = messages \cup requestVoteMessages
    /\ UNCHANGED <<log, commitIndex, nextIndex, matchIndex, heartbeatElapsed, 
                   randomizedElectionTimeout, uncommittedSize, appliedIndex, keyValueStore,
                   clientRequests, clientResponses, readStates, pendingReadIndex, isLearner,
                   recentActive, inflightMsgs>>

HandleRequestVote(s, m) ==
    /\ m.type = "RequestVote"
    /\ m.to = s
    /\ LET grant == /\ m.term >= currentTerm[s]
                    /\ CanVote(s, m.from, m.term, m.index, m.logTerm)
       IN /\ IF m.term > currentTerm[s]
             THEN BecomeFollower(s, m.term, None)
             ELSE UNCHANGED <<state, currentTerm, leader, votedFor, electionElapsed, 
                             heartbeatElapsed, votes, leadTransferee>>
          /\ IF grant
             THEN /\ votedFor' = [votedFor EXCEPT ![s] = m.from]
                  /\ electionElapsed' = [electionElapsed EXCEPT ![s] = 0]
             ELSE UNCHANGED <<votedFor, electionElapsed>>
          /\ messages' = (messages \ {m}) \cup 
                        {[type |-> "RequestVoteResponse", from |-> s, to |-> m.from, 
                          term |-> currentTerm'[s], logTerm |-> 0, index |-> 0,
                          entries |-> <<>>, commit |-> 0, success |-> grant, 
                          data |-> "", context |-> ""]}
          /\ UNCHANGED <<log, commitIndex, nextIndex, matchIndex, randomizedElectionTimeout,
                         uncommittedSize, appliedIndex, keyValueStore, clientRequests,
                         clientResponses, readStates, pendingReadIndex, isLearner,
                         recentActive, inflightMsgs, pendingConfIndex>>

HandleRequestVoteResponse(s, m) ==
    /\ m.type = "RequestVoteResponse"
    /\ m.to = s
    /\ state[s] = "Candidate"
    /\ m.term = currentTerm[s]
    /\ IF m.success
       THEN /\ votes' = [votes EXCEPT ![s] = votes[s] \cup {m.from}]
            /\ IF votes'[s] \in Quorum
               THEN /\ BecomeLeader(s)
                    /\ LET heartbeatMessages ==
                           {[type |-> "Heartbeat", from |-> s, to |-> t, term |-> currentTerm'[s],
                             logTerm |-> 0, index |-> 0, entries |-> <<>>, 
                             commit |-> commitIndex[s], success |-> FALSE, data |-> "", context |-> ""]
                            : t \in Servers \ {s}}
                       IN messages' = (messages \ {m}) \cup heartbeatMessages
               ELSE /\ messages' = messages \ {m}
                    /\ UNCHANGED <<state, currentTerm, leader, nextIndex, matchIndex, 
                                   heartbeatElapsed, pendingConfIndex>>
       ELSE /\ votes' = votes
            /\ messages' = messages \ {m}
            /\ UNCHANGED <<state, currentTerm, leader, nextIndex, matchIndex, 
                           heartbeatElapsed, pendingConfIndex>>
    /\ UNCHANGED <<log, commitIndex, votedFor, electionElapsed, randomizedElectionTimeout,
                   uncommittedSize, appliedIndex, keyValueStore, clientRequests,
                   clientResponses, readStates, pendingReadIndex, leadTransferee,
                   isLearner, recentActive, inflightMsgs>>

AppendEntries(s, t) ==
    /\ state[s] = "Leader"
    /\ t \in Servers \ {s}
    /\ inflightMsgs[s][t] < MaxInflightMsgs
    /\ LET prevIndex == nextIndex[s][t] - 1
           prevTerm == LogTerm(log[s], prevIndex)
           entries == SubSeq(log[s], nextIndex[s][t], 
                             Min(Len(log[s]), nextIndex[s][t] + MaxCommittedSizePerReady - 1))
       IN /\ messages' = messages \cup 
                        {[type |-> "AppendEntries", from |-> s, to |-> t, term |-> currentTerm[s],
                          logTerm |-> prevTerm, index |-> prevIndex, entries |-> entries,
                          commit |-> commitIndex[s], success |-> FALSE, data |-> "", context |-> ""]}
          /\ inflightMsgs' = [inflightMsgs EXCEPT ![s][t] = inflightMsgs[s][t] + 1]
    /\ UNCHANGED <<currentTerm, votedFor, log, commitIndex, state, leader, nextIndex, matchIndex,
                   votes, electionElapsed, heartbeatElapsed, randomizedElectionTimeout,
                   uncommittedSize, appliedIndex, keyValueStore, clientRequests,
                   clientResponses, readStates, pendingReadIndex, leadTransferee,
                   isLearner, recentActive, pendingConfIndex>>

HandleAppendEntries(s, m) ==
    /\ m.type = "AppendEntries"
    /\ m.to = s
    /\ IF m.term > currentTerm[s]
       THEN BecomeFollower(s, m.term, m.from)
       ELSE UNCHANGED <<state, currentTerm, leader, votedFor, electionElapsed, 
                        heartbeatElapsed, votes, leadTransferee>>
    /\ LET success == /\ m.term >= currentTerm[s]
                      /\ \/ m.index = 0
                         \/ /\ m.index <= Len(log[s])
                            /\ LogTerm(log[s], m.