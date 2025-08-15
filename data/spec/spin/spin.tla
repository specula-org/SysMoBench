---- MODULE spin ----

EXTENDS Naturals, Sequences, TLC

CONSTANTS Threads

VARIABLES 
    lock_state,
    thread_state,
    critical_section,
    pc

vars == <<lock_state, thread_state, critical_section, pc>>

ThreadStates == {"idle", "trying", "spinning", "locked"}
ProgramCounters == {"idle", "try_acquire", "spin_loop", "locked", "unlock"}

TypeOK == 
    /\ lock_state \in BOOLEAN
    /\ thread_state \in [Threads -> ThreadStates]
    /\ critical_section \in [Threads -> BOOLEAN]
    /\ pc \in [Threads -> ProgramCounters]

Init ==
    /\ lock_state = FALSE
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ critical_section = [t \in Threads |-> FALSE]
    /\ pc = [t \in Threads |-> "idle"]

StartLock(t) ==
    /\ pc[t] = "idle"
    /\ thread_state[t] = "idle"
    /\ pc' = [pc EXCEPT ![t] = "try_acquire"]
    /\ thread_state' = [thread_state EXCEPT ![t] = "trying"]
    /\ UNCHANGED <<lock_state, critical_section>>

TryAcquire(t) ==
    /\ pc[t] = "try_acquire"
    /\ thread_state[t] = "trying"
    /\ IF lock_state = FALSE
       THEN /\ lock_state' = TRUE
            /\ pc' = [pc EXCEPT ![t] = "locked"]
            /\ thread_state' = [thread_state EXCEPT ![t] = "locked"]
            /\ critical_section' = [critical_section EXCEPT ![t] = TRUE]
       ELSE /\ pc' = [pc EXCEPT ![t] = "spin_loop"]
            /\ thread_state' = [thread_state EXCEPT ![t] = "spinning"]
            /\ UNCHANGED <<lock_state, critical_section>>

SpinLoop(t) ==
    /\ pc[t] = "spin_loop"
    /\ thread_state[t] = "spinning"
    /\ pc' = [pc EXCEPT ![t] = "try_acquire"]
    /\ thread_state' = [thread_state EXCEPT ![t] = "trying"]
    /\ UNCHANGED <<lock_state, critical_section>>

TryLock(t) ==
    /\ pc[t] = "idle"
    /\ thread_state[t] = "idle"
    /\ IF lock_state = FALSE
       THEN /\ lock_state' = TRUE
            /\ pc' = [pc EXCEPT ![t] = "locked"]
            /\ thread_state' = [thread_state EXCEPT ![t] = "locked"]
            /\ critical_section' = [critical_section EXCEPT ![t] = TRUE]
       ELSE /\ pc' = [pc EXCEPT ![t] = "idle"]
            /\ UNCHANGED <<lock_state, thread_state, critical_section>>

Unlock(t) ==
    /\ pc[t] = "locked"
    /\ thread_state[t] = "locked"
    /\ critical_section[t] = TRUE
    /\ lock_state = TRUE
    /\ lock_state' = FALSE
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
    /\ critical_section' = [critical_section EXCEPT ![t] = FALSE]

Next == 
    \E t \in Threads:
        \/ StartLock(t)
        \/ TryAcquire(t)
        \/ SpinLoop(t)
        \/ TryLock(t)
        \/ Unlock(t)

Spec == Init /\ [][Next]_vars

====