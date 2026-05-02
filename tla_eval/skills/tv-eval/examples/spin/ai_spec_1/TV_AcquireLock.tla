---- MODULE TV_AcquireLock ----
\* Window Validator for AcquireLock. Reads windows from JSON at runtime.
\* Per-invocation: WINDOW_INDEX env var selects the window.

EXTENDS Naturals, FiniteSets, TLC, Json, IOUtils

CONSTANTS Threads

VARIABLES lockState, pc, requestType, step

vars == <<lockState, pc, requestType, step>>

S == INSTANCE spin

AllWindows == ndJsonDeserialize("windows_AcquireLock.ndjson")
w == AllWindows[atoi(IOEnv.WINDOW_INDEX)]

Init ==
    /\ lockState = w.pre.lockState
    /\ pc = w.pre.pc
    /\ requestType \in [Threads -> {"blocking", "nonblocking", "none"}]
    /\ step = 0

Next ==
    /\ step = 0
    /\ \E t \in Threads : S!AcquireLock(t)
    /\ step' = 1

PostReached ==
    /\ step = 1
    /\ lockState = w.post.lockState
    /\ pc = w.post.pc

NeverPost == ~PostReached

Spec == Init /\ [][Next]_vars

====
