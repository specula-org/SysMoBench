"""
Tests for the TLA+ backend's cfg-handling helpers.

The strip helper protects the phase-4 invariant check from being contaminated
by model-supplied INVARIANT / PROPERTY / CONSTRAINT / ACTION_CONSTRAINT /
POSTCONDITION sections in the base cfg forwarded from phase 2.
"""

import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

from tla_eval.languages.tla_plus import _strip_user_supplied_assertions


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TLC_JAR = _PROJECT_ROOT / "lib" / "tla2tools.jar"
# Behavioral tests need both the jar and a JVM on PATH; gate on both so a
# missing `java` skips rather than errors.
_TLC_AVAILABLE = _TLC_JAR.exists() and shutil.which("java") is not None
_TLC_SKIP_REASON = (
    "TLC unavailable: need lib/tla2tools.jar (run `sysmobench-setup`) and `java` on PATH."
)


def test_strip_one_liner_invariant_and_property():
    """One-liner INVARIANT / PROPERTY entries are removed from the base cfg."""
    cfg = dedent("""\
        SPECIFICATION Spec
        CONSTANT Max = 3

        INVARIANT ModelTypeOK
        INVARIANT ModelNoOverflow
        PROPERTY ModelTermination
        """)
    out = _strip_user_supplied_assertions(cfg)

    assert "ModelTypeOK" not in out
    assert "ModelNoOverflow" not in out
    assert "ModelTermination" not in out
    # Keepers survive.
    assert "SPECIFICATION Spec" in out
    assert "CONSTANT Max = 3" in out


def test_strip_block_form_invariants_and_properties():
    """Block-form INVARIANTS / PROPERTIES with indented continuation lines."""
    cfg = dedent("""\
        SPECIFICATION Spec
        CONSTANT Nodes = {1, 2, 3}

        INVARIANTS
            ModelOneLeader
            ModelLogConsistency

        PROPERTIES
            ModelEventualConsistency
        """)
    out = _strip_user_supplied_assertions(cfg)

    assert "ModelOneLeader" not in out
    assert "ModelLogConsistency" not in out
    assert "ModelEventualConsistency" not in out
    assert "SPECIFICATION Spec" in out
    assert "Nodes = {1, 2, 3}" in out


def test_strip_constraint_and_action_constraint():
    """CONSTRAINT / ACTION_CONSTRAINT shrink the state space and must be stripped."""
    cfg = dedent("""\
        SPECIFICATION Spec
        CONSTANT Bound = 5

        CONSTRAINT ModelStateBound
        ACTION_CONSTRAINT ModelStepBound
        """)
    out = _strip_user_supplied_assertions(cfg)

    assert "ModelStateBound" not in out
    assert "ModelStepBound" not in out
    assert "CONSTRAINT" not in out
    assert "ACTION_CONSTRAINT" not in out
    assert "SPECIFICATION Spec" in out
    assert "CONSTANT Bound = 5" in out


def test_strip_plural_constraint_keywords():
    """CONSTRAINTS / ACTION_CONSTRAINTS (plural) are valid TLC keywords too —
    a model emitting the plural form must not slip past the strip."""
    cfg = dedent("""\
        SPECIFICATION Spec
        CONSTRAINTS
            ModelStateBoundA
            ModelStateBoundB
        ACTION_CONSTRAINTS
            ModelStepBoundA
        """)
    out = _strip_user_supplied_assertions(cfg)

    assert "ModelStateBoundA" not in out
    assert "ModelStateBoundB" not in out
    assert "ModelStepBoundA" not in out
    assert "CONSTRAINTS" not in out
    assert "ACTION_CONSTRAINTS" not in out
    assert "SPECIFICATION Spec" in out


def test_strip_postcondition():
    """POSTCONDITION is in the same risk class as INVARIANT and is stripped."""
    cfg = dedent("""\
        SPECIFICATION Spec
        POSTCONDITION ModelEndCheck
        """)
    out = _strip_user_supplied_assertions(cfg)

    assert "ModelEndCheck" not in out
    assert "POSTCONDITION" not in out


def test_strip_plural_postcondition():
    """POSTCONDITIONS (plural) is a valid TLC keyword and must be stripped in
    both one-liner and block form, like the other plural variants."""
    one_liner = _strip_user_supplied_assertions(
        "SPECIFICATION Spec\nPOSTCONDITIONS ModelPost\n"
    )
    assert "ModelPost" not in one_liner
    assert "POSTCONDITIONS" not in one_liner

    block = _strip_user_supplied_assertions(
        "SPECIFICATION Spec\nPOSTCONDITIONS\n    ModelPostA\n    ModelPostB\n"
    )
    assert "ModelPostA" not in block
    assert "ModelPostB" not in block
    assert "POSTCONDITIONS" not in block
    assert "SPECIFICATION Spec" in block


def test_strip_inline_block_comment_before_keyword():
    """A `(* ... *)` block comment before a section keyword on the same line
    must not hide the keyword: TLC honors `(* c *) INVARIANT Foo`, so the strip
    has to as well or model assertions leak past it."""
    cfg = dedent("""\
        SPECIFICATION Spec
        (* model note *) INVARIANT ModelHaltsAt5
        (* x *) CONSTRAINT ModelBound
        """)
    out = _strip_user_supplied_assertions(cfg)

    assert "ModelHaltsAt5" not in out
    assert "ModelBound" not in out
    assert "INVARIANT" not in out
    assert "CONSTRAINT" not in out
    assert "SPECIFICATION Spec" in out


def test_block_comment_keyword_is_not_a_section_header():
    """A section keyword that appears only inside a `(* ... *)` block comment is
    commentary, not a header, and must not flip the parser into drop mode and
    swallow following kept content."""
    cfg = dedent("""\
        SPECIFICATION Spec
        (* note: the INVARIANT below is intentional
           and PROPERTY handling is documented here *)
        CONSTANT Max = 3
        """)
    out = _strip_user_supplied_assertions(cfg)

    assert "SPECIFICATION Spec" in out
    assert "CONSTANT Max = 3" in out


def test_comment_inside_dropped_section_does_not_leak():
    """A comment that annotates a stripped section is dropped with it, instead
    of dangling, detached, in the output."""
    cfg = dedent("""\
        SPECIFICATION Spec
        INVARIANT
            \\* the model invariant list
            ModelInv
        CONSTANT Max = 3
        """)
    out = _strip_user_supplied_assertions(cfg)

    assert "ModelInv" not in out
    assert "the model invariant list" not in out
    assert "SPECIFICATION Spec" in out
    assert "CONSTANT Max = 3" in out


def test_keyword_named_constant_binding_is_preserved():
    """A constant whose name collides with an assertion keyword (e.g. a binding
    `PROPERTY = TRUE` under CONSTANT) is an assignment, not a section header,
    and the binding (plus following bindings) must survive."""
    cfg = dedent("""\
        CONSTANT
            N = 3
            PROPERTY = TRUE
            M = 5
        INVARIANT <- SomeOp
        """)
    out = _strip_user_supplied_assertions(cfg)

    assert "N = 3" in out
    assert "PROPERTY = TRUE" in out
    assert "M = 5" in out
    assert "INVARIANT <- SomeOp" in out


def test_crlf_input_is_normalized():
    """CRLF line endings are normalized so kept lines do not carry a stray
    carriage return into the composed cfg."""
    out = _strip_user_supplied_assertions(
        "SPECIFICATION Spec\r\nCONSTANT Max = 3\r\nINVARIANT ModelInv\r\n"
    )
    assert "\r" not in out
    assert "ModelInv" not in out
    assert "SPECIFICATION Spec" in out


def test_preserves_non_assertion_sections():
    """Non-assertion sections survive: SPECIFICATION, INIT, NEXT, CONSTANT(S),
    SYMMETRY, VIEW, ALIAS, CHECK_DEADLOCK, plus comments and blank lines."""
    cfg = dedent("""\
        SPECIFICATION Spec
        INIT InitState
        NEXT NextRel
        CONSTANT
            N = 3
            Values = {0, 1}
        SYMMETRY SymNodes
        VIEW StateView
        ALIAS AliasName
        CHECK_DEADLOCK FALSE

        \\* This comment must survive
        INVARIANT ModelToBeDropped
        """)
    out = _strip_user_supplied_assertions(cfg)

    # Kept.
    assert "SPECIFICATION Spec" in out
    assert "INIT InitState" in out
    assert "NEXT NextRel" in out
    assert "N = 3" in out
    assert "Values = {0, 1}" in out
    assert "SYMMETRY SymNodes" in out
    assert "VIEW StateView" in out
    assert "ALIAS AliasName" in out
    assert "CHECK_DEADLOCK FALSE" in out
    assert "This comment must survive" in out
    # Dropped.
    assert "ModelToBeDropped" not in out


def test_section_boundary_recovers_into_keep():
    """A KEEP section header immediately following a DROP section must end the drop."""
    cfg = dedent("""\
        INVARIANT ModelOne

        SPECIFICATION Spec
        CONSTANT Max = 3
        """)
    out = _strip_user_supplied_assertions(cfg)

    assert "ModelOne" not in out
    assert "SPECIFICATION Spec" in out
    assert "CONSTANT Max = 3" in out


def test_idempotent():
    """Re-stripping a stripped cfg is a no-op."""
    cfg = dedent("""\
        SPECIFICATION Spec
        CONSTANT Max = 3
        """)
    once = _strip_user_supplied_assertions(cfg)
    twice = _strip_user_supplied_assertions(once)

    assert once == twice


def test_final_cfg_sent_to_tlc_contains_only_expert_invariant():
    """End-to-end: a contaminated base cfg, run through the strip and the
    per-invariant composition path, yields a cfg that contains only the
    expert invariant (plus the scaffolding we keep)."""
    from tla_eval.evaluation.semantics.manual_invariant_evaluator import (
        StaticConfigGenerator,
    )

    contaminated = dedent("""\
        SPECIFICATION Spec
        CONSTANT Max = 3
        INVARIANT ModelInvariant
        PROPERTY ModelProperty
        CONSTRAINT ModelConstraint
        ACTION_CONSTRAINT ModelActionConstraint
        POSTCONDITION ModelPost
        """)

    stripped = _strip_user_supplied_assertions(contaminated)
    gen = StaticConfigGenerator()
    ok, final_cfg, err = gen.generate_config_for_invariant_from_base(
        stripped, "ExpertInvariant", "safety"
    )

    assert ok, err
    # Expert invariant lands.
    assert "ExpertInvariant" in final_cfg
    # Every model-supplied assertion/constraint is gone.
    assert "ModelInvariant" not in final_cfg
    assert "ModelProperty" not in final_cfg
    assert "ModelConstraint" not in final_cfg
    assert "ModelActionConstraint" not in final_cfg
    assert "ModelPost" not in final_cfg
    # Scaffolding survives.
    assert "SPECIFICATION Spec" in final_cfg
    assert "CONSTANT Max = 3" in final_cfg


def test_compose_liveness_property_path_after_strip():
    """The liveness path composes into a PROPERTY section. Stripping a model
    PROPERTY first, then adding the expert one, must leave only the expert."""
    from tla_eval.evaluation.semantics.manual_invariant_evaluator import (
        StaticConfigGenerator,
    )

    stripped = _strip_user_supplied_assertions(
        "SPECIFICATION Spec\nCONSTANT Max = 3\nPROPERTY ModelLiveness\n"
    )
    gen = StaticConfigGenerator()
    ok, final_cfg, err = gen.generate_config_for_invariant_from_base(
        stripped, "ExpertLiveness", "liveness"
    )

    assert ok, err
    assert "ExpertLiveness" in final_cfg
    assert "ModelLiveness" not in final_cfg
    assert "SPECIFICATION Spec" in final_cfg


def test_compose_from_fully_stripped_base():
    """A base cfg that was entirely model assertions strips to almost nothing;
    composition must still produce a valid single expert-invariant section."""
    from tla_eval.evaluation.semantics.manual_invariant_evaluator import (
        StaticConfigGenerator,
    )

    stripped = _strip_user_supplied_assertions(
        "INVARIANT ModelA\nPROPERTY ModelB\n"
    )
    gen = StaticConfigGenerator()
    ok, final_cfg, err = gen.generate_config_for_invariant_from_base(
        stripped, "ExpertInv", "safety"
    )

    assert ok, err
    assert "ExpertInv" in final_cfg
    assert "ModelA" not in final_cfg
    assert "ModelB" not in final_cfg


# ---------------------------------------------------------------------------
# Behavioral integration tests: prove the strip actually changes TLC's verdict
# on the same spec, not just the text of the cfg. Gated on TLC being present.
# ---------------------------------------------------------------------------


_COUNTER_SPEC = dedent(
    """\
    ---- MODULE Counter ----
    EXTENDS Naturals
    VARIABLE x

    Init == x = 0
    Next == /\\ x < 20
            /\\ x' = x + 1
    Spec == Init /\\ [][Next]_x

    \\* Expert invariant: actually true throughout the reachable state space.
    ExpertOK == x \\in Nat

    \\* Model-supplied "halting" invariant: false at x=5.
    ModelHaltsAt5 == x < 5

    \\* Model-supplied state constraint: caps exploration at x in [0,3].
    ModelBound == x <= 3

    \\* Expert invariant that should be violated when x reaches 10.
    ExpertFailsAt10 == x < 10
    ====
    """
)


def _run_tlc(work_dir: Path, spec_name: str, cfg_name: str, timeout: int = 60):
    return subprocess.run(
        [
            "java",
            "-cp",
            str(_TLC_JAR),
            "tlc2.TLC",
            "-workers",
            "1",
            "-config",
            cfg_name,
            spec_name,
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _compose(base_cfg: str, invariant_name: str) -> str:
    """Mirror the production flow: hand a base cfg to StaticConfigGenerator
    and ask it to add the expert invariant. Used by the behavioral tests to
    build the cfg that would actually reach TLC."""
    from tla_eval.evaluation.semantics.manual_invariant_evaluator import (
        StaticConfigGenerator,
    )

    gen = StaticConfigGenerator()
    ok, final_cfg, err = gen.generate_config_for_invariant_from_base(
        base_cfg, invariant_name, "safety"
    )
    assert ok, err
    return final_cfg


@pytest.mark.skipif(not _TLC_AVAILABLE, reason=_TLC_SKIP_REASON)
def test_strip_changes_tlc_verdict_for_invariant_contamination(tmp_path):
    """A model-supplied INVARIANT carried in the base cfg ends up alongside
    the expert invariant in the cfg sent to TLC. If the model's invariant
    is the first to be violated, TLC halts and the per-template parser
    attributes the violation to the expert invariant's run — a false FAIL.
    Stripping the base cfg before composition removes the contamination."""
    (tmp_path / "Counter.tla").write_text(_COUNTER_SPEC)

    base = dedent(
        """\
        SPECIFICATION Spec
        INVARIANT ModelHaltsAt5
        """
    )

    contaminated_final = _compose(base, "ExpertOK")
    clean_final = _compose(_strip_user_supplied_assertions(base), "ExpertOK")

    # Sanity: contaminated cfg carries the model's invariant; clean one doesn't.
    assert "ModelHaltsAt5" in contaminated_final
    assert "ExpertOK" in contaminated_final
    assert "ModelHaltsAt5" not in clean_final
    assert "ExpertOK" in clean_final

    (tmp_path / "contam.cfg").write_text(contaminated_final)
    (tmp_path / "clean.cfg").write_text(clean_final)

    contam_run = _run_tlc(tmp_path, "Counter", "contam.cfg")
    clean_run = _run_tlc(tmp_path, "Counter", "clean.cfg")

    # Contaminated: TLC halts on the model's invariant before it can issue a
    # clean verdict on ExpertOK. Violation message names ModelHaltsAt5.
    assert contam_run.returncode != 0
    assert "Invariant ModelHaltsAt5 is violated" in contam_run.stdout
    assert "Invariant ExpertOK is violated" not in contam_run.stdout

    # Clean: TLC explores the full reachable space, ExpertOK holds throughout.
    assert "is violated" not in clean_run.stdout


@pytest.mark.skipif(not _TLC_AVAILABLE, reason=_TLC_SKIP_REASON)
def test_strip_changes_tlc_verdict_for_constraint_contamination(tmp_path):
    """A model-supplied CONSTRAINT carried in the base cfg shrinks the state
    space TLC explores. If the constraint excludes the states that would
    falsify the expert invariant, the invariant "passes" — a false PASS.
    Stripping the base cfg restores full coverage."""
    (tmp_path / "Counter.tla").write_text(_COUNTER_SPEC)

    base = dedent(
        """\
        SPECIFICATION Spec
        CONSTRAINT ModelBound
        """
    )

    contaminated_final = _compose(base, "ExpertFailsAt10")
    clean_final = _compose(_strip_user_supplied_assertions(base), "ExpertFailsAt10")

    # Sanity: CONSTRAINT survives in contaminated, gone in clean.
    assert "ModelBound" in contaminated_final
    assert "CONSTRAINT" in contaminated_final
    assert "ModelBound" not in clean_final
    assert "CONSTRAINT" not in clean_final

    (tmp_path / "contam.cfg").write_text(contaminated_final)
    (tmp_path / "clean.cfg").write_text(clean_final)

    contam_run = _run_tlc(tmp_path, "Counter", "contam.cfg")
    clean_run = _run_tlc(tmp_path, "Counter", "clean.cfg")

    # Contaminated: ModelBound caps x at 3, TLC never reaches x=10,
    # ExpertFailsAt10 "passes" — false negative.
    assert "is violated" not in contam_run.stdout

    # Clean: TLC reaches x=10 and catches the real violation.
    assert "Invariant ExpertFailsAt10 is violated" in clean_run.stdout


@pytest.mark.skipif(not _TLC_AVAILABLE, reason=_TLC_SKIP_REASON)
def test_strip_changes_tlc_verdict_for_block_comment_invariant(tmp_path):
    """TLC honors a section keyword preceded by an inline `(* ... *)` block
    comment, so a model can carry one in its base cfg. Left unstripped it
    contaminates the expert run (false FAIL); the comment-aware strip removes
    it."""
    (tmp_path / "Counter.tla").write_text(_COUNTER_SPEC)

    base = dedent(
        """\
        SPECIFICATION Spec
        (* model note *) INVARIANT ModelHaltsAt5
        """
    )

    contaminated_final = _compose(base, "ExpertOK")
    clean_final = _compose(_strip_user_supplied_assertions(base), "ExpertOK")

    assert "ModelHaltsAt5" in contaminated_final
    assert "ModelHaltsAt5" not in clean_final
    assert "ExpertOK" in clean_final

    (tmp_path / "contam.cfg").write_text(contaminated_final)
    (tmp_path / "clean.cfg").write_text(clean_final)

    contam_run = _run_tlc(tmp_path, "Counter", "contam.cfg")
    clean_run = _run_tlc(tmp_path, "Counter", "clean.cfg")

    # Contaminated: TLC parses and enforces the commented-prefixed model
    # invariant, halting before ExpertOK gets a clean verdict.
    assert "Invariant ModelHaltsAt5 is violated" in contam_run.stdout
    # Clean: only ExpertOK remains, and it holds throughout.
    assert "is violated" not in clean_run.stdout
