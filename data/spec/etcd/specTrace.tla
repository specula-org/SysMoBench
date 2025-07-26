--------------------------- MODULE specTrace ---------------------------

EXTENDS TLC, Sequences, SequencesExt, Naturals, FiniteSets, Bags, Json, IOUtils, etcdraft, TraceSpec, TVOperators


TraceNil == "null"

(* Extract system configuration from first trace line *)
TraceServers == ToSet(Trace[1].Servers)
TraceMaxTerm == ToSet(Trace[1].MaxTerm)
TraceMaxLogLen == ToSet(Trace[1].MaxLogLen)
TraceMaxCommittedEntries == ToSet(Trace[1].MaxCommittedEntries)

(* Default variable initialization *)
DefaultImpl(varName) ==
    CASE varName = "currentTerm" -> [s \in TraceServers |-> 0]
     [] varName = "votedFor" -> [s \in TraceServers |-> Nil]
     [] varName = "log" -> [s \in TraceServers |-> <<>>]
     [] varName = "commitIndex" -> [s \in TraceServers |-> 0]
     [] varName = "state" -> [s \in TraceServers |-> "Follower"]
     [] varName = "nextIndex" -> [s \in TraceServers |-> [t \in TraceServers |-> 1]]
     [] varName = "matchIndex" -> [s \in TraceServers |-> [t \in TraceServers |-> 0]]
     [] varName = "votes" -> [s \in TraceServers |-> {}]
     [] varName = "leader" -> [s \in TraceServers |-> Nil]
     [] varName = "electionTimeout" -> [s \in TraceServers |-> 5]
     [] varName = "heartbeatTimeout" -> [s \in TraceServers |-> 1]
     [] varName = "messages" -> {}
     [] varName = "clientRequests" -> {}
     [] varName = "appliedIndex" -> [s \in TraceServers |-> 0]
     [] varName = "keyValueStore" -> [s \in TraceServers |-> <<>>]

(* State variable update logic *)
UpdateVariablesImpl(t) ==
    /\ IF "currentTerm" \in DOMAIN t
       THEN currentTerm' = UpdateVariable(currentTerm, "currentTerm", t)
       ELSE TRUE
    /\ IF "votedFor" \in DOMAIN t
       THEN votedFor' = UpdateVariable(votedFor, "votedFor", t)
       ELSE TRUE
    /\ IF "log" \in DOMAIN t
       THEN log' = UpdateVariable(log, "log", t)
       ELSE TRUE
    /\ IF "commitIndex" \in DOMAIN t
       THEN commitIndex' = UpdateVariable(commitIndex, "commitIndex", t)
       ELSE TRUE
    /\ IF "state" \in DOMAIN t
       THEN state' = UpdateVariable(state, "state", t)
       ELSE TRUE
    /\ IF "nextIndex" \in DOMAIN t
       THEN nextIndex' = UpdateVariable(nextIndex, "nextIndex", t)
       ELSE TRUE
    /\ IF "matchIndex" \in DOMAIN t
       THEN matchIndex' = UpdateVariable(matchIndex, "matchIndex", t)
       ELSE TRUE
    /\ IF "votes" \in DOMAIN t
       THEN votes' = UpdateVariable(votes, "votes", t)
       ELSE TRUE
    /\ IF "leader" \in DOMAIN t
       THEN leader' = UpdateVariable(leader, "leader", t)
       ELSE TRUE
    /\ IF "electionTimeout" \in DOMAIN t
       THEN electionTimeout' = UpdateVariable(electionTimeout, "electionTimeout", t)
       ELSE TRUE
    /\ IF "heartbeatTimeout" \in DOMAIN t
       THEN heartbeatTimeout' = UpdateVariable(heartbeatTimeout, "heartbeatTimeout", t)
       ELSE TRUE
    /\ IF "messages" \in DOMAIN t
       THEN messages' = UpdateVariable(messages, "messages", t)
       ELSE TRUE
    /\ IF "clientRequests" \in DOMAIN t
       THEN clientRequests' = UpdateVariable(clientRequests, "clientRequests", t)
       ELSE TRUE
    /\ IF "appliedIndex" \in DOMAIN t
       THEN appliedIndex' = UpdateVariable(appliedIndex, "appliedIndex", t)
       ELSE TRUE
    /\ IF "keyValueStore" \in DOMAIN t
       THEN keyValueStore' = UpdateVariable(keyValueStore, "keyValueStore", t)
       ELSE TRUE

(* Action event matching *)

IsTimeout ==
    /\ IsEvent("Timeout")
    /\ \E s \in TraceServers :
        Timeout(s)

IsTickElectionTimeout ==
    /\ IsEvent("TickElectionTimeout")
    /\ \E s \in TraceServers :
        TickElectionTimeout(s)

IsTickHeartbeatTimeout ==
    /\ IsEvent("TickHeartbeatTimeout")
    /\ \E s \in TraceServers :
        TickHeartbeatTimeout(s)

IsHeartbeat ==
    /\ IsEvent("Heartbeat")
    /\ \E s \in TraceServers :
        Heartbeat(s)

IsUpdateCommitIndex ==
    /\ IsEvent("UpdateCommitIndex")
    /\ \E s \in TraceServers :
        UpdateCommitIndex(s)

IsApplyEntry ==
    /\ IsEvent("ApplyEntry")
    /\ \E s \in TraceServers :
        ApplyEntry(s)

IsRequestVote ==
    /\ IsEvent("RequestVote")
    /\ \E s \in TraceServers :
        /\ \E t \in TraceServers \ {s} :
            RequestVote(s, t)

IsAppendEntries ==
    /\ IsEvent("AppendEntries")
    /\ \E s \in TraceServers :
        /\ \E t \in TraceServers \ {s} :
            AppendEntries(s, t)

IsClientRequest ==
    /\ IsEvent("ClientRequest")
    /\ \E s \in TraceServers :
        /\ \E req \in clientRequests :
            ClientRequest(s, req)

IsReceiveMessage ==
    /\ IsEvent("ReceiveMessage")
    /\ \E m \in messages :
        ReceiveMessage(m)

(* State transition definition *)
TraceNextImpl ==
    \/ IsTimeout
    \/ IsTickElectionTimeout
    \/ IsTickHeartbeatTimeout
    \/ IsHeartbeat
    \/ IsUpdateCommitIndex
    \/ IsApplyEntry
    \/ IsRequestVote
    \/ IsAppendEntries
    \/ IsClientRequest
    \/ IsReceiveMessage


(* REPLACE / MODIFY COMMENT BELOW ONLY IF YOU WANT TO MAKE ACTION COMPOSITION *)
ComposedNext == FALSE

(* NOTHING TO CHANGE BELOW *)
BaseSpec == Init /\ [][Next \/ ComposedNext]_vars

=============================================================================

