---- MODULE TV_Windows ----
EXTENDS Naturals, FiniteSets, TLC, Json, IOUtils

CONSTANTS Threads
VARIABLES lockState, pc, step
vars == <<lockState, pc, step>>
S == INSTANCE spin

AllWindows == ndJsonDeserialize("action_windows.jsonl")
w == AllWindows[atoi(IOEnv.WINDOW_INDEX)]
actor == atoi(w.actor) + 1
PC(s) == [t \in Threads |-> s.pc[ToString(t - 1)]]

Init ==
    /\ lockState = w.pre_state.lockState
    /\ pc = PC(w.pre_state)
    /\ step = 0

Next ==
    /\ step = 0
    /\ CASE w.action \in {"TryAcquireBlocking", "TryAcquireNonBlocking"} -> S!TryAcquire(actor)
         [] w.action = "AcquireSuccess" -> (S!AcquireLock(actor) \/ S!AcquireLockDirect(actor))
         [] w.action = "AcquireFail" -> S!AcquireFail(actor)
         [] w.action = "Release" -> S!ReleaseLock(actor)
         [] OTHER -> FALSE
    /\ step' = 1

PostReached == step = 1 /\ lockState = w.post_state.lockState /\ pc = PC(w.post_state)
NeverPost == ~PostReached
Spec == Init /\ [][Next]_vars
====
