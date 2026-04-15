---- MODULE etcdraft ----
EXTENDS TLC, Sequences, SequencesExt, Naturals, FiniteSets, Bags

\* Constants
CONSTANTS
    Nodes,          \* Set of node IDs
    MaxTerm,        \* Maximum term number
    MaxLogLen       \* Maximum log length

\* Node states
CONSTANTS
    StateFollower,
    StateCandidate,
    StateLeader,
    StatePreCandidate

\* Message types
CONSTANTS
    MsgHup,
    MsgVote,
    MsgVoteResp,
    MsgPreVote,
    MsgPreVoteResp,
    MsgApp,
    MsgAppResp,
    MsgHeartbeat,
    MsgHeartbeatResp,
    MsgProp,
    MsgBeat

ASSUME Nodes /= {}
ASSUME MaxTerm > 0
ASSUME MaxLogLen > 0

\* Special values
CONSTANTS None

\* Variables
VARIABLES
    \* Persistent state on all nodes
    currentTerm,    \* Current term for each node
    votedFor,       \* Candidate voted for in current term
    log,            \* Log entries (sequence of records)

    \* Volatile state on all nodes
    state,          \* Current state (Follower/Candidate/Leader/PreCandidate)
    commitIndex,    \* Index of highest log entry known to be committed

    \* Volatile state on leaders
    leader,         \* Current leader (None if unknown)

    \* Election timing
    electionElapsed,    \* Election timeout counter
    heartbeatElapsed,   \* Heartbeat timeout counter

    \* Network
    messages        \* In-flight messages

vars == <<currentTerm, votedFor, log, state, commitIndex, leader,
          electionElapsed, heartbeatElapsed, messages>>

----

\* Helper functions

\* Majority quorum check
Quorum == {S \in SUBSET Nodes : Cardinality(S) * 2 > Cardinality(Nodes)}

IsMajority(S) == S \in Quorum

\* Last log entry helpers
LastLogIndex(n) == Len(log[n])

LastLogTerm(n) ==
    IF Len(log[n]) = 0 THEN 0 ELSE log[n][Len(log[n])].term

\* Log matching check - candidate's log is at least as up-to-date
IsLogUpToDate(n, candidateTerm, candidateIndex) ==
    LET lastTerm == LastLogTerm(n)
        lastIndex == LastLogIndex(n)
    IN \/ candidateTerm > lastTerm
       \/ /\ candidateTerm = lastTerm
          /\ candidateIndex >= lastIndex

\* Get term at a specific index
LogTerm(n, index) ==
    IF index = 0 \/ index > Len(log[n]) THEN 0
    ELSE log[n][index].term

\* Check if logs match at a given index and term (for MsgApp)
LogMatches(n, prevIndex, prevTerm) ==
    \/ prevIndex = 0
    \/ /\ prevIndex <= Len(log[n])
       /\ LogTerm(n, prevIndex) = prevTerm

\* Send a message
Send(m) == messages' = messages \cup {m}

\* Discard a message
Discard(m) == messages' = messages \ {m}

\* Reply to a message with a new message
Reply(response, request) ==
    messages' = (messages \ {request}) \cup {response}

----

\* Initial state

Init ==
    /\ currentTerm = [n \in Nodes |-> 0]
    /\ votedFor = [n \in Nodes |-> None]
    /\ log = [n \in Nodes |-> <<>>]
    /\ state = [n \in Nodes |-> StateFollower]
    /\ commitIndex = [n \in Nodes |-> 0]
    /\ leader = [n \in Nodes |-> None]
    /\ electionElapsed = [n \in Nodes |-> 0]
    /\ heartbeatElapsed = [n \in Nodes |-> 0]
    /\ messages = {}

----

\* Election timeout - node starts an election
ElectionTimeout(n) ==
    /\ state[n] \in {StateFollower, StateCandidate, StatePreCandidate}
    /\ electionElapsed[n] >= 1  \* Simplified: any positive value triggers timeout
    /\ state' = [state EXCEPT ![n] = StatePreCandidate]  \* PreVote is enabled by default
    /\ leader' = [leader EXCEPT ![n] = None]
    /\ electionElapsed' = [electionElapsed EXCEPT ![n] = 0]
    /\ LET term == currentTerm[n] + 1  \* PreVote uses next term
           lastIndex == LastLogIndex(n)
           lastTerm == LastLogTerm(n)
       IN /\ messages' = messages \cup
                {[type |-> MsgPreVote,
                  from |-> n,
                  to |-> m,
                  term |-> term,
                  logIndex |-> lastIndex,
                  logTerm |-> lastTerm] : m \in Nodes \ {n}}
    /\ UNCHANGED <<currentTerm, votedFor, log, commitIndex, heartbeatElapsed>>

\* PreCandidate receives enough votes and becomes Candidate
BecomeCandidate(n) ==
    /\ state[n] = StatePreCandidate
    /\ LET prevotes == {m \in messages :
                          /\ m.type = MsgPreVoteResp
                          /\ m.to = n
                          /\ m.term = currentTerm[n] + 1
                          /\ m.voteGranted}
           voters == {m.from : m \in prevotes} \cup {n}
       IN IsMajority(voters)
    /\ currentTerm' = [currentTerm EXCEPT ![n] = currentTerm[n] + 1]
    /\ votedFor' = [votedFor EXCEPT ![n] = n]
    /\ state' = [state EXCEPT ![n] = StateCandidate]
    /\ electionElapsed' = [electionElapsed EXCEPT ![n] = 0]
    /\ LET term == currentTerm[n] + 1
           lastIndex == LastLogIndex(n)
           lastTerm == LastLogTerm(n)
       IN messages' = messages \cup
            {[type |-> MsgVote,
              from |-> n,
              to |-> m,
              term |-> term,
              logIndex |-> lastIndex,
              logTerm |-> lastTerm] : m \in Nodes \ {n}}
    /\ UNCHANGED <<log, commitIndex, leader, heartbeatElapsed>>

\* Candidate receives enough votes and becomes Leader
BecomeLeader(n) ==
    /\ state[n] = StateCandidate
    /\ LET votes == {m \in messages :
                       /\ m.type = MsgVoteResp
                       /\ m.to = n
                       /\ m.term = currentTerm[n]
                       /\ m.voteGranted}
           voters == {m.from : m \in votes} \cup {n}
       IN IsMajority(voters)
    /\ state' = [state EXCEPT ![n] = StateLeader]
    /\ leader' = [leader EXCEPT ![n] = n]
    /\ heartbeatElapsed' = [heartbeatElapsed EXCEPT ![n] = 0]
    /\ electionElapsed' = [electionElapsed EXCEPT ![n] = 0]
    /\ LET emptyEntry == [term |-> currentTerm[n], data |-> "empty"]
       IN log' = [log EXCEPT ![n] = Append(@, emptyEntry)]
    /\ messages' = messages \cup
         {[type |-> MsgApp,
           from |-> n,
           to |-> m,
           term |-> currentTerm[n],
           prevIndex |-> LastLogIndex(n),
           prevTerm |-> LastLogTerm(n),
           entries |-> <<>>,
           commitIndex |-> commitIndex[n]] : m \in Nodes \ {n}}
    /\ UNCHANGED <<currentTerm, votedFor, commitIndex>>

----

\* Handle PreVote request
HandlePreVoteRequest(m) ==
    /\ m.type = MsgPreVote
    /\ LET n == m.to
           canVote == IsLogUpToDate(n, m.logTerm, m.logIndex)
           grant == canVote
       IN /\ Reply([type |-> MsgPreVoteResp,
                    from |-> n,
                    to |-> m.from,
                    term |-> m.term,
                    voteGranted |-> grant], m)
          /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                        leader, electionElapsed, heartbeatElapsed>>

\* Handle Vote request
HandleVoteRequest(m) ==
    /\ m.type = MsgVote
    /\ LET n == m.to
       IN /\ m.term >= currentTerm[n]
          /\ \/ m.term > currentTerm[n]
             \/ /\ m.term = currentTerm[n]
                /\ votedFor[n] \in {None, m.from}
          /\ LET canVote == IsLogUpToDate(n, m.logTerm, m.logIndex)
                 grant == canVote
                 newTerm == IF m.term > currentTerm[n] THEN m.term ELSE currentTerm[n]
             IN /\ currentTerm' = [currentTerm EXCEPT ![n] = newTerm]
                /\ votedFor' = [votedFor EXCEPT ![n] = IF grant THEN m.from ELSE @]
                /\ state' = [state EXCEPT ![n] =
                              IF m.term > currentTerm[n] THEN StateFollower ELSE @]
                /\ leader' = [leader EXCEPT ![n] =
                              IF m.term > currentTerm[n] THEN None ELSE @]
                /\ Reply([type |-> MsgVoteResp,
                         from |-> n,
                         to |-> m.from,
                         term |-> newTerm,
                         voteGranted |-> grant], m)
                /\ UNCHANGED <<log, commitIndex, electionElapsed, heartbeatElapsed>>

\* Handle Vote rejection for old term
RejectOldVote(m) ==
    /\ m.type = MsgVote
    /\ m.term < currentTerm[m.to]
    /\ Reply([type |-> MsgVoteResp,
             from |-> m.to,
             to |-> m.from,
             term |-> currentTerm[m.to],
             voteGranted |-> FALSE], m)
    /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                  leader, electionElapsed, heartbeatElapsed>>

----

\* Leader sends heartbeat
LeaderHeartbeat(n) ==
    /\ state[n] = StateLeader
    /\ heartbeatElapsed[n] >= 1  \* Simplified timeout check
    /\ heartbeatElapsed' = [heartbeatElapsed EXCEPT ![n] = 0]
    /\ messages' = messages \cup
         {[type |-> MsgHeartbeat,
           from |-> n,
           to |-> m,
           term |-> currentTerm[n],
           commitIndex |-> commitIndex[n]] : m \in Nodes \ {n}}
    /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                  leader, electionElapsed>>

\* Handle Heartbeat from leader
HandleHeartbeat(m) ==
    /\ m.type = MsgHeartbeat
    /\ LET n == m.to
       IN /\ m.term >= currentTerm[n]
          /\ currentTerm' = [currentTerm EXCEPT ![n] = m.term]
          /\ state' = [state EXCEPT ![n] = StateFollower]
          /\ leader' = [leader EXCEPT ![n] = m.from]
          /\ electionElapsed' = [electionElapsed EXCEPT ![n] = 0]
          /\ commitIndex' = [commitIndex EXCEPT ![n] = m.commitIndex]
          /\ Reply([type |-> MsgHeartbeatResp,
                   from |-> n,
                   to |-> m.from,
                   term |-> m.term], m)
          /\ UNCHANGED <<votedFor, log, heartbeatElapsed>>

----

\* Client proposes a new entry to the leader
ClientProposal(n) ==
    /\ state[n] = StateLeader
    /\ Len(log[n]) < MaxLogLen
    /\ LET newEntry == [term |-> currentTerm[n], data |-> "proposal"]
       IN /\ log' = [log EXCEPT ![n] = Append(@, newEntry)]
          /\ UNCHANGED <<currentTerm, votedFor, state, commitIndex,
                        leader, electionElapsed, heartbeatElapsed, messages>>

\* Leader sends AppendEntries to replicate log
LeaderAppendEntries(n, m) ==
    /\ state[n] = StateLeader
    /\ m \in Nodes \ {n}
    /\ LET prevIndex == Len(log[n]) - 1
           prevTerm == IF prevIndex = 0 THEN 0 ELSE log[n][prevIndex].term
           entries == IF Len(log[n]) > 0 THEN <<log[n][Len(log[n])]>> ELSE <<>>
       IN /\ messages' = messages \cup
               {[type |-> MsgApp,
                 from |-> n,
                 to |-> m,
                 term |-> currentTerm[n],
                 prevIndex |-> prevIndex,
                 prevTerm |-> prevTerm,
                 entries |-> entries,
                 commitIndex |-> commitIndex[n]]}
          /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                        leader, electionElapsed, heartbeatElapsed>>

\* Handle AppendEntries request
HandleAppendEntries(m) ==
    /\ m.type = MsgApp
    /\ LET n == m.to
       IN /\ m.term >= currentTerm[n]
          /\ currentTerm' = [currentTerm EXCEPT ![n] = m.term]
          /\ state' = [state EXCEPT ![n] = StateFollower]
          /\ leader' = [leader EXCEPT ![n] = m.from]
          /\ electionElapsed' = [electionElapsed EXCEPT ![n] = 0]
          /\ IF LogMatches(n, m.prevIndex, m.prevTerm)
             THEN \* Log matches, append entries
                  LET newLog == SubSeq(log[n], 1, m.prevIndex) \o m.entries
                  IN /\ log' = [log EXCEPT ![n] = newLog]
                     /\ commitIndex' = [commitIndex EXCEPT ![n] = m.commitIndex]
                     /\ Reply([type |-> MsgAppResp,
                              from |-> n,
                              to |-> m.from,
                              term |-> m.term,
                              success |-> TRUE,
                              matchIndex |-> m.prevIndex + Len(m.entries)], m)
             ELSE \* Log doesn't match, reject
                  /\ Reply([type |-> MsgAppResp,
                           from |-> n,
                           to |-> m.from,
                           term |-> m.term,
                           success |-> FALSE,
                           matchIndex |-> 0], m)
                  /\ UNCHANGED <<log, commitIndex>>
          /\ UNCHANGED <<votedFor, heartbeatElapsed>>

\* Leader handles AppendEntries response
HandleAppendEntriesResponse(m) ==
    /\ m.type = MsgAppResp
    /\ LET n == m.to
       IN /\ state[n] = StateLeader
          /\ m.term = currentTerm[n]
          /\ IF m.success
             THEN \* Success - try to commit
                  LET matchIndices == {i \in 1..Len(log[n]) :
                                        \E quorum \in Quorum :
                                          /\ n \in quorum
                                          /\ \A node \in quorum :
                                             \/ node = n
                                             \/ \E msg \in messages :
                                                /\ msg.type = MsgAppResp
                                                /\ msg.from = node
                                                /\ msg.to = n
                                                /\ msg.success
                                                /\ msg.matchIndex >= i}
                      maxCommit == IF matchIndices = {} THEN commitIndex[n]
                                   ELSE CHOOSE i \in matchIndices :
                                        \A j \in matchIndices : i >= j
                      newCommit == IF maxCommit > commitIndex[n]
                                     /\ LogTerm(n, maxCommit) = currentTerm[n]
                                   THEN maxCommit
                                   ELSE commitIndex[n]
                  IN /\ commitIndex' = [commitIndex EXCEPT ![n] = newCommit]
                     /\ Discard(m)
             ELSE \* Failure - retry needed (simplified: just discard)
                  /\ Discard(m)
                  /\ UNCHANGED commitIndex
          /\ UNCHANGED <<currentTerm, votedFor, log, state, leader,
                        electionElapsed, heartbeatElapsed>>

\* Reject old AppendEntries
RejectOldAppendEntries(m) ==
    /\ m.type = MsgApp
    /\ m.term < currentTerm[m.to]
    /\ Reply([type |-> MsgAppResp,
             from |-> m.to,
             to |-> m.from,
             term |-> currentTerm[m.to],
             success |-> FALSE,
             matchIndex |-> 0], m)
    /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                  leader, electionElapsed, heartbeatElapsed>>

----

\* Step down to follower when receiving higher term
StepDown(n, newTerm) ==
    /\ newTerm > currentTerm[n]
    /\ currentTerm' = [currentTerm EXCEPT ![n] = newTerm]
    /\ state' = [state EXCEPT ![n] = StateFollower]
    /\ votedFor' = [votedFor EXCEPT ![n] = None]
    /\ leader' = [leader EXCEPT ![n] = None]
    /\ UNCHANGED <<log, commitIndex, electionElapsed, heartbeatElapsed, messages>>

\* Advance time (election and heartbeat timers)
Tick(n) ==
    /\ electionElapsed' = [electionElapsed EXCEPT ![n] = @ + 1]
    /\ heartbeatElapsed' = [heartbeatElapsed EXCEPT ![n] = @ + 1]
    /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                  leader, messages>>

----

\* Type invariant
TypeOK ==
    /\ currentTerm \in [Nodes -> 0..MaxTerm]
    /\ votedFor \in [Nodes -> Nodes \cup {None}]
    /\ log \in [Nodes -> Seq([term : 0..MaxTerm, data : STRING])]
    /\ state \in [Nodes -> {StateFollower, StateCandidate, StateLeader, StatePreCandidate}]
    /\ commitIndex \in [Nodes -> 0..MaxLogLen]
    /\ leader \in [Nodes -> Nodes \cup {None}]
    /\ electionElapsed \in [Nodes -> Nat]
    /\ heartbeatElapsed \in [Nodes -> Nat]
    /\ messages \subseteq [type : {MsgHup, MsgVote, MsgVoteResp, MsgPreVote,
                                   MsgPreVoteResp, MsgApp, MsgAppResp,
                                   MsgHeartbeat, MsgHeartbeatResp, MsgProp, MsgBeat},
                          from : Nodes,
                          to : Nodes,
                          term : 0..MaxTerm,
                          logIndex : 0..MaxLogLen,
                          logTerm : 0..MaxTerm,
                          voteGranted : BOOLEAN,
                          prevIndex : 0..MaxLogLen,
                          prevTerm : 0..MaxTerm,
                          entries : Seq([term : 0..MaxTerm, data : STRING]),
                          commitIndex : 0..MaxLogLen,
                          success : BOOLEAN,
                          matchIndex : 0..MaxLogLen]

----

\* Next state relation
Next ==
    \/ \E n \in Nodes : ElectionTimeout(n)
    \/ \E n \in Nodes : BecomeCandidate(n)
    \/ \E n \in Nodes : BecomeLeader(n)
    \/ \E m \in messages : HandlePreVoteRequest(m)
    \/ \E m \in messages : HandleVoteRequest(m)
    \/ \E m \in messages : RejectOldVote(m)
    \/ \E n \in Nodes : LeaderHeartbeat(n)
    \/ \E m \in messages : HandleHeartbeat(m)
    \/ \E n \in Nodes : ClientProposal(n)
    \/ \E n, m \in Nodes : LeaderAppendEntries(n, m)
    \/ \E m \in messages : HandleAppendEntries(m)
    \/ \E m \in messages : HandleAppendEntriesResponse(m)
    \/ \E m \in messages : RejectOldAppendEntries(m)
    \/ \E n \in Nodes : Tick(n)

\* Specification with fairness
Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ \A n \in Nodes : WF_vars(Tick(n))
    /\ \A n \in Nodes : WF_vars(ElectionTimeout(n))
    /\ \A n \in Nodes : WF_vars(BecomeCandidate(n))
    /\ \A n \in Nodes : WF_vars(BecomeLeader(n))
    /\ \A n \in Nodes : WF_vars(LeaderHeartbeat(n))
    /\ \A m \in messages : WF_vars(HandleHeartbeat(m))
    /\ \A m \in messages : WF_vars(HandleVoteRequest(m))
    /\ \A m \in messages : WF_vars(HandlePreVoteRequest(m))

====
