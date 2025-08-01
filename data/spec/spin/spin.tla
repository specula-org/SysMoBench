---- MODULE spin ----
EXTENDS Naturals

CONSTANTS Threads

VARIABLES lock, pc

Init == 
    lock = FALSE
    /\ pc = [t \in Threads |-> "start"]

Start(thread) == 
    /\ pc[thread] = "start"
    /\ pc' = [pc EXCEPT ![thread] = "trying"]
    /\ lock' = lock

TryAcquireSucceeds(thread) ==
    /\ pc[thread] = "trying"
    /\ lock = FALSE
    /\ lock' = TRUE
    /\ pc' = [pc EXCEPT ![thread] = "critical"]

TryAcquireFails(thread) ==
    /\ pc[thread] = "trying"
    /\ lock = TRUE
    /\ lock' = lock
    /\ pc' = [pc EXCEPT ![thread] = "trying"]

Release(thread) ==
    /\ pc[thread] = "critical"
    /\ lock' = FALSE
    /\ pc' = [pc EXCEPT ![thread] = "start"]

Next == 
    \E thread \in Threads: 
        Start(thread) 
        \/ TryAcquireSucceeds(thread)
        \/ TryAcquireFails(thread)
        \/ Release(thread)

MutualExclusion == 
    \A t1, t2 \in Threads: 
        t1 # t2 => ~ (pc[t1] = "critical" /\ pc[t2] = "critical")

LockCorrect == 
    lock = ( \E t \in Threads: pc[t] = "critical" )

Invariant == 
    /\ \A t \in Threads: pc[t] \in {"start", "trying", "critical"}
    /\ MutualExclusion
    /\ LockCorrect

====