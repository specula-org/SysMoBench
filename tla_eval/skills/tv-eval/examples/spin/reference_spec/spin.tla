---- MODULE spin ----
EXTENDS Naturals, FiniteSets

CONSTANTS Threads

VARIABLES lockState, pc

vars == <<lockState, pc>>

TypeOK ==
    /\ lockState \in {"locked", "unlocked"}
    /\ pc \in [Threads -> {"idle", "acquiring", "locked"}]

Init ==
    /\ lockState = "unlocked"
    /\ pc = [t \in Threads |-> "idle"]

\* Thread starts a blocking acquire attempt (lock is held, will spin)
TryAcquire(t) ==
    /\ pc[t] = "idle"
    /\ lockState = "locked"
    /\ pc' = [pc EXCEPT ![t] = "acquiring"]
    /\ UNCHANGED lockState

\* Thread acquires lock after spinning (lock just became free)
AcquireLock(t) ==
    /\ pc[t] = "acquiring"
    /\ lockState = "unlocked"
    /\ lockState' = "locked"
    /\ pc' = [pc EXCEPT ![t] = "locked"]

\* Thread acquires lock immediately (no contention, lock is free)
AcquireLockDirect(t) ==
    /\ pc[t] = "idle"
    /\ lockState = "unlocked"
    /\ lockState' = "locked"
    /\ pc' = [pc EXCEPT ![t] = "locked"]

\* Thread releases lock
ReleaseLock(t) ==
    /\ pc[t] = "locked"
    /\ lockState' = "unlocked"
    /\ pc' = [pc EXCEPT ![t] = "idle"]

Next ==
    \E t \in Threads:
        \/ TryAcquire(t)
        \/ AcquireLock(t)
        \/ AcquireLockDirect(t)
        \/ ReleaseLock(t)

Spec == Init /\ [][Next]_vars

MutualExclusion == Cardinality({t \in Threads : pc[t] = "locked"}) <= 1

====
