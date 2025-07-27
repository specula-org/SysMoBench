---- MODULE etcdraft ----
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS Servers,        \* The set of servers
          Values,         \* The set of possible values
          Keys,           \* The set of possible keys
          MaxTerm,        \* Maximum term number
          MaxIndex,       \* Maximum log index
          NoServer,       \* Null value for server ID
          NoValue,        \* Null value for values
          NoKey           \* Null value for keys

VARIABLES \* Server state
          currentTerm,    \* Current term of each server
          state,          \* State of each server (Follower, Candidate, Leader)
          votedFor,       \* Server that received vote in current term (or NoServer)
          log,            \* Log of each server: sequence of records
          commitIndex,    \* Index of highest log entry known to be committed
          lastApplied,    \* Index of highest log entry applied to state machine
          
          \* Leader state
          nextIndex,      \* For each server, index of next log entry to send
          matchIndex,     \* For each server, index of highest log entry known to be replicated
          leaderId,       \* Current leader or NoServer if unknown
          
          \* Candidate state
          votesGranted,   \* Set of servers that granted vote to candidate
          
          \* Network state
          messages,       \* Set of in-flight messages
          
          \* Application state
          kvStore,        \* Key-value store state for each server
          clientRequests, \* Set of pending client requests
          clientResponses \* Set of responses sent to clients

serverVars == <<currentTerm, state, votedFor>>
leaderVars == <<nextIndex, matchIndex, leaderId>>
candidateVars == <<votesGranted>>
logVars == <<log, commitIndex, lastApplied>>
networkVars == <<messages>>
appVars == <<kvStore, clientRequests, clientResponses>>

vars == <<serverVars, leaderVars, candidateVars, logVars, networkVars, appVars>>

\* Message types
TypeAppendEntries == "AppendEntries"
TypeAppendEntriesResponse == "AppendEntriesResponse"
TypeRequestVote == "RequestVote"
TypeRequestVoteResponse == "RequestVoteResponse"
TypeClientRequest == "ClientRequest"
TypeClientResponse == "ClientResponse"
TypePreVote == "PreVote"
TypePreVoteResponse == "PreVoteResponse"
TypeTimeoutNow == "TimeoutNow"
TypeReadIndex == "ReadIndex"
TypeReadIndexResponse == "ReadIndexResponse"

\* Server states
Follower == "Follower"
Candidate == "Candidate"
Leader == "Leader"
PreCandidate == "PreCandidate"

\* Client operation types
OpGet == "Get"
OpPut == "Put"
OpDelete == "Delete"

\* Entry types
EntryNormal == "EntryNormal"
EntryConfChange == "EntryConfChange"

--------------------------------------------------------------------
\* Helper functions

\* Return the minimum value from a set
Min(s) == CHOOSE x \in s : \A y \in s : x <= y

\* Return the maximum value from a set
Max(s) == CHOOSE x \in s : \A y \in s : x >= y

\* Return the maximum of two values
max(a, b) == IF a > b THEN a ELSE b

\* Return the minimum of two values
min(a, b) == IF a < b THEN a ELSE b

\* Is the log entry at the given index the same term in both logs?
LogTermMatches(i, term, xLog) ==
    /\ i > 0
    /\ i <= Len(xLog)
    /\ xLog[i].term = term

\* Is the log entry at the given index present in the log?
EntryExists(i, xLog) ==
    /\ i > 0
    /\ i <= Len(xLog)

\* Is the candidate's log at least as up-to-date as the voter's?
IsUpToDate(candidateLastIndex, candidateLastTerm, voterLastIndex, voterLastTerm) ==
    \/ candidateLastTerm > voterLastTerm
    \/ /\ candidateLastTerm = voterLastTerm
       /\ candidateLastIndex >= voterLastIndex

\* Return the term of the last entry in the log, or 0 if the log is empty
LastTerm(xLog) ==
    IF Len(xLog) = 0 THEN 0 ELSE xLog[Len(xLog)].term

\* Return the index of the last entry in the log, or 0 if the log is empty
LastIndex(xLog) ==
    Len(xLog)

\* Find the index of the conflicting entry in the log
FindConflictByTerm(index, term, xLog) ==
    LET possibleIndices == {i \in 1..min(index, Len(xLog)) : xLog[i].term <= term}
    IN IF possibleIndices = {} THEN 0
       ELSE Max(possibleIndices)

--------------------------------------------------------------------
\* Initial state

InitServerVars ==
    /\ currentTerm = [s \in Servers |-> 0]
    /\ state = [s \in Servers |-> Follower]
    /\ votedFor = [s \in Servers |-> NoServer]

InitLeaderVars ==
    /\ nextIndex = [s \in Servers |-> [t \in Servers |-> 1]]
    /\ matchIndex = [s \in Servers |-> [t \in Servers |-> 0]]
    /\ leaderId = NoServer

InitCandidateVars ==
    /\ votesGranted = [s \in Servers |-> {}]

InitLogVars ==
    /\ log = [s \in Servers |-> <<>>]
    /\ commitIndex = [s \in Servers |-> 0]
    /\ lastApplied = [s \in Servers |-> 0]

InitNetworkVars ==
    /\ messages = {}

InitAppVars ==
    /\ kvStore = [s \in Servers |-> [k \in Keys |-> NoValue]]
    /\ clientRequests = {}
    /\ clientResponses = {}

Init ==
    /\ InitServerVars
    /\ InitLeaderVars
    /\ InitCandidateVars
    /\ InitLogVars
    /\ InitNetworkVars
    /\ InitAppVars

--------------------------------------------------------------------
\* Message sending and receiving

\* Add a message to the network
Send(m) ==
    messages' = messages \union {m}

\* Remove a message from the network
Discard(m) ==
    messages' = messages \ {m}

\* Combination of Send and Discard
SendAndDiscard(send, discard) ==
    messages' = (messages \union {send}) \ {discard}

--------------------------------------------------------------------
\* Server actions

\* Server i times out and becomes a pre-candidate if pre-vote is enabled
BecomePreCandidate(i) ==
    /\ state[i] \in {Follower, Candidate}
    /\ state' = [state EXCEPT ![i] = PreCandidate]
    /\ votesGranted' = [votesGranted EXCEPT ![i] = {i}]
    /\ leaderId' = [leaderId EXCEPT ![i] = NoServer]
    /\ LET lastIdx == LastIndex(log[i])
           lastTrm == LastTerm(log[i])
           newTerm == currentTerm[i] + 1
       IN /\ Send([mtype         |-> TypePreVote,
                   mterm         |-> newTerm,
                   msource       |-> i,
                   mdest         |-> i,
                   mlastLogTerm  |-> lastTrm,
                   mlastLogIndex |-> lastIdx])
          /\ \/ /\ i = leaderId[i]
                /\ leaderId' = [leaderId EXCEPT ![i] = NoServer]
             \/ /\ i # leaderId[i]
                /\ UNCHANGED leaderId
    /\ UNCHANGED <<currentTerm, votedFor, logVars, nextIndex, matchIndex, appVars>>

\* Server i times out and starts a new election
BecomeCandidate(i) ==
    /\ state[i] \in {Follower, Candidate, PreCandidate}
    /\ state' = [state EXCEPT ![i] = Candidate]
    /\ currentTerm' = [currentTerm EXCEPT ![i] = currentTerm[i] + 1]
    /\ votedFor' = [votedFor EXCEPT ![i] = i]
    /\ votesGranted' = [votesGranted EXCEPT ![i] = {i}]
    /\ leaderId' = [leaderId EXCEPT ![i] = NoServer]
    /\ LET lastIdx == LastIndex(log[i])
           lastTrm == LastTerm(log[i])
       IN \/ /\ state[i] = PreCandidate
             /\ \A j \in Servers \ {i} :
                  Send([mtype         |-> TypeRequestVote,
                        mterm         |-> currentTerm'[i],
                        msource       |-> i,
                        mdest         |-> j,
                        mlastLogTerm  |-> lastTrm,
                        mlastLogIndex |-> lastIdx])
          \/ /\ state[i] # PreCandidate
             /\ \A j \in Servers \ {i} :
                  Send([mtype         |-> TypeRequestVote,
                        mterm         |-> currentTerm'[i],
                        msource       |-> i,
                        mdest         |-> j,
                        mlastLogTerm  |-> lastTrm,
                        mlastLogIndex |-> lastIdx])
    /\ UNCHANGED <<logVars, nextIndex, matchIndex, appVars>>

\* Server i sends a PreVote request to server j
SendPreVoteRequest(i, j) ==
    /\ state[i] = PreCandidate
    /\ LET lastIdx == LastIndex(log[i])
           lastTrm == LastTerm(log[i])
           newTerm == currentTerm[i] + 1
       IN Send([mtype         |-> TypePreVote,
                mterm         |-> newTerm,
                msource       |-> i,
                mdest         |-> j,
                mlastLogTerm  |-> lastTrm,
                mlastLogIndex |-> lastIdx])
    /\ UNCHANGED <<serverVars, leaderVars, candidateVars, logVars, appVars>>

\* Server i receives a PreVote request from server j
HandlePreVoteRequest(i, j, m) ==
    LET logOk == \/ m.mlastLogTerm > LastTerm(log[i])
                 \/ /\ m.mlastLogTerm = LastTerm(log[i])
                    /\ m.mlastLogIndex >= LastIndex(log[i])
        grant == /\ m.mterm > currentTerm[i]
                 /\ logOk
                 /\ votedFor[i] = NoServer \/ votedFor[i] = j
    IN /\ m.mtype = TypePreVote
       /\ \/ /\ grant
             /\ Send([mtype        |-> TypePreVoteResponse,
                      mterm        |-> m.mterm,
                      msource      |-> i,
                      mdest        |-> j,
                      mvoteGranted |-> TRUE])
          \/ /\ ~grant
             /\ Send([mtype        |-> TypePreVoteResponse,
                      mterm        |-> currentTerm[i],
                      msource      |-> i,
                      mdest        |-> j,
                      mvoteGranted |-> FALSE])
       /\ Discard(m)
       /\ UNCHANGED <<serverVars, leaderVars, candidateVars, logVars, appVars>>

\* Server i receives a PreVote response from server j
HandlePreVoteResponse(i, j, m) ==
    /\ m.mtype = TypePreVoteResponse
    /\ state[i] = PreCandidate
    /\ \/ /\ m.mvoteGranted
          /\ votesGranted' = [votesGranted EXCEPT ![i] = votesGranted[i] \union {j}]
          /\ LET quorum == (Cardinality(Servers) \div 2) + 1
             IN \/ /\ Cardinality(votesGranted'[i]) >= quorum
                   /\ BecomeCandidate(i)
                \/ /\ Cardinality(votesGranted'[i]) < quorum
                   /\ UNCHANGED <<serverVars, leaderVars, logVars, appVars>>
       \/ /\ ~m.mvoteGranted
          /\ \/ /\ m.mterm > currentTerm[i]
                /\ currentTerm' = [currentTerm EXCEPT ![i] = m.mterm]
                /\ state' = [state EXCEPT ![i] = Follower]
                /\ votedFor' = [votedFor EXCEPT ![i] = NoServer]
                /\ UNCHANGED <<leaderVars, candidateVars, logVars, appVars>>
             \/ /\ m.mterm <= currentTerm[i]
                /\ UNCHANGED <<serverVars, leaderVars, candidateVars, logVars, appVars>>
    /\ Discard(m)

\* Server i receives a RequestVote request from server j
HandleRequestVoteRequest(i, j, m) ==
    LET logOk == \/ m.mlastLogTerm > LastTerm(log[i])
                 \/ /\ m.mlastLogTerm = LastTerm(log[i])
                    /\ m.mlastLogIndex >= LastIndex(log[i])
        grant == /\ m.mterm = currentTerm[i]
                 /\ logOk
                 /\ votedFor[i] \in {NoServer, j}
    IN /\ m.mtype = TypeRequestVote
       /\ \/ /\ m.mterm > currentTerm[i]
             /\ currentTerm' = [currentTerm EXCEPT ![i] = m.mterm]
             /\ state' = [state EXCEPT ![i] = Follower]
             /\ votedFor' = [votedFor EXCEPT ![i] = NoServer]
             /\ LET logOkUpdated == \/ m.mlastLogTerm > LastTerm(log[i])
                                    \/ /\ m.mlastLogTerm = LastTerm(log[i])
                                       /\ m.mlastLogIndex >= LastIndex(log[i])
                    grantUpdated == /\ logOkUpdated
                                    /\ votedFor'[i] \in {NoServer, j}
                IN \/ /\ grantUpdated
                      /\ votedFor' = [votedFor EXCEPT ![i] = j]
                      /\ Send([mtype        |-> TypeRequestVoteResponse,
                               mterm        |-> m.mterm,
                               msource      |-> i,
                               mdest        |-> j,
                               mvoteGranted |-> TRUE])
                   \/ /\ ~grantUpdated
                      /\ Send([mtype        |-> TypeRequestVoteResponse,
                               mterm        |-> m.mterm,
                               msource      |-> i,
                               mdest        |-> j,
                               mvoteGranted |-> FALSE])
          \/ /\ m.mterm = currentTerm[i]
             /\ \/ /\ grant
                   /\ votedFor' = [votedFor EXCEPT ![i] = j]
                   /\ Send([mtype        |-> TypeRequestVoteResponse,
                            mterm        |-> currentTerm[i],
                            msource      |-> i,
                            mdest        |-> j,
                            mvoteGranted |-> TRUE])
                \/ /\ ~grant
                   /\ Send([mtype        |-> TypeRequestVoteResponse,
                            mterm        |-> currentTerm[i],
                            msource      |-> i,
                            mdest        |-> j,
                            mvoteGranted |-> FALSE])