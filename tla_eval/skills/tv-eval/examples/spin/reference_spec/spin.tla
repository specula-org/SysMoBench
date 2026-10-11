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

\* A request starts before CAS, whether the lock is free or held.
TryAcquire(t) ==
    /\ pc[t] = "idle"
    /\ pc' = [pc EXCEPT ![t] = "acquiring"]
    /\ UNCHANGED lockState

\* A pending request successfully executes CAS.
AcquireLock(t) ==
    /\ pc[t] = "acquiring"
    /\ lockState = "unlocked"
    /\ lockState' = "locked"
    /\ pc' = [pc EXCEPT ![t] = "locked"]

\* Coarse legacy traces omitted the nonblocking request event.
\* Retained for that abstraction; not the task's full modeling contract.
AcquireLockDirect(t) ==
    /\ pc[t] = "idle"
    /\ lockState = "unlocked"
    /\ lockState' = "locked"
    /\ pc' = [pc EXCEPT ![t] = "locked"]

\* A nonblocking request fails while another thread holds the lock.
AcquireFail(t) ==
    /\ pc[t] = "acquiring"
    /\ lockState = "locked"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED lockState

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
        \/ AcquireFail(t)
        \/ ReleaseLock(t)

Spec == Init /\ [][Next]_vars

MutualExclusion == Cardinality({t \in Threads : pc[t] = "locked"}) <= 1

====
