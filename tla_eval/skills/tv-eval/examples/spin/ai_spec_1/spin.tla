---- MODULE spin ----
EXTENDS TLC, Sequences, SequencesExt, Naturals, FiniteSets, Bags

CONSTANTS Threads

VARIABLES lockState, pc, requestType

Vars == <<lockState, pc, requestType>>

TypeOK ==
  /\ lockState \in {"locked", "unlocked"}
  /\ pc \in [Threads -> {"idle", "trying", "spinning", "locked"}]
  /\ requestType \in [Threads -> {"blocking", "nonblocking", "none"}]

Init ==
  /\ lockState = "unlocked"
  /\ pc = [t \in Threads |-> "idle"]
  /\ requestType = [t \in Threads |-> "none"]

RequestBlocking(t) ==
  /\ pc[t] = "idle"
  /\ pc' = [pc EXCEPT ![t] = "trying"]
  /\ requestType' = [requestType EXCEPT ![t] = "blocking"]
  /\ UNCHANGED <<lockState>>

RequestNonBlocking(t) ==
  /\ pc[t] = "idle"
  /\ pc' = [pc EXCEPT ![t] = "trying"]
  /\ requestType' = [requestType EXCEPT ![t] = "nonblocking"]
  /\ UNCHANGED <<lockState>>

AcquireLock(t) ==
  /\ pc[t] = "trying"
  /\ lockState = "unlocked"
  /\ lockState' = "locked"
  /\ pc' = [pc EXCEPT ![t] = "locked"]
  /\ UNCHANGED <<requestType>>

AcquireFail(t) ==
  /\ pc[t] = "trying"
  /\ lockState = "locked"
  /\ requestType[t] = "nonblocking"
  /\ pc' = [pc EXCEPT ![t] = "idle"]
  /\ UNCHANGED <<lockState, requestType>>

EnterSpinning(t) ==
  /\ pc[t] = "trying"
  /\ lockState = "locked"
  /\ requestType[t] = "blocking"
  /\ pc' = [pc EXCEPT ![t] = "spinning"]
  /\ UNCHANGED <<lockState, requestType>>

SpinLoop(t) ==
  /\ pc[t] = "spinning"
  /\ pc' = [pc EXCEPT ![t] = "trying"]
  /\ UNCHANGED <<lockState, requestType>>

ReleaseLock(t) ==
  /\ pc[t] = "locked"
  /\ lockState' = "unlocked"
  /\ pc' = [pc EXCEPT ![t] = "idle"]
  /\ requestType' = [requestType EXCEPT ![t] = "none"]

ThreadNext(t) ==
  \/ RequestBlocking(t)
  \/ RequestNonBlocking(t)
  \/ AcquireLock(t)
  \/ AcquireFail(t)
  \/ EnterSpinning(t)
  \/ SpinLoop(t)
  \/ ReleaseLock(t)

Next ==
  \E t \in Threads : ThreadNext(t)

Spec == Init /\ [][Next]_Vars /\ \A t \in Threads : WF_Vars(ThreadNext(t))

====
