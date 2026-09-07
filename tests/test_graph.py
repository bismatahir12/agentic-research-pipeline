"""Unit tests focused on the parts of the system that don't require a live
LLM call: schema validation and the critic routing logic. Full end-to-end
graph runs are better suited to an integration test gated behind a real
ANTHROPIC_API_KEY, since they cost money and hit the network.
"""

from src.graph import _route_after_critic
from src.schemas import CritiqueOutput, PlanOutput, ReportOutput


def test_plan_output_schema_accepts_valid_data():
    plan = PlanOutput(subtasks=["What is X?", "How does Y compare to X?"])
    assert len(plan.subtasks) == 2


def test_report_output_schema_requires_fields():
    report = ReportOutput(
        title="Test Report",
        executive_summary="Summary.",
        sections=["## Section 1\nBody text."],
    )
    assert report.title == "Test Report"


def test_route_after_critic_approved_goes_to_finalize():
    state = {
        "critique": CritiqueOutput(approved=True, score=9, feedback=""),
        "revision_count": 0,
    }
    assert _route_after_critic(state) == "finalize"


def test_route_after_critic_rejected_under_limit_goes_to_revise():
    state = {
        "critique": CritiqueOutput(approved=False, score=4, feedback="Too thin."),
        "revision_count": 0,
    }
    assert _route_after_critic(state) == "revise"


def test_route_after_critic_rejected_at_limit_finalizes_anyway():
    state = {
        "critique": CritiqueOutput(approved=False, score=4, feedback="Still thin."),
        "revision_count": 2,  # equals MAX_REVISIONS default
    }
    assert _route_after_critic(state) == "finalize"


def test_critique_score_bounds_are_enforced():
    try:
        CritiqueOutput(approved=True, score=11, feedback="oops")
        assert False, "score above 10 should raise a validation error"
    except Exception:
        pass


def test_objective_checks_catch_empty_sections():
    from src.agents.critic import _objective_checks

    report = ReportOutput(
        title="T", executive_summary="Fine summary here.", sections=[""]
    )
    failures = _objective_checks(report)
    assert any("empty" in f for f in failures)


def test_objective_checks_catch_short_report():
    from src.agents.critic import _objective_checks

    report = ReportOutput(
        title="T", executive_summary="Too short.", sections=["## A\nShort."]
    )
    failures = _objective_checks(report)
    assert any("words" in f for f in failures)


def test_objective_checks_pass_for_solid_report():
    from src.agents.critic import _objective_checks

    long_section = "## Findings\n" + ("This is a real sentence. " * 30)
    report = ReportOutput(
        title="T",
        executive_summary="A reasonably detailed executive summary of the findings.",
        sections=[long_section],
    )
    assert _objective_checks(report) == []


def test_finalize_node_documents_revision_limit_hit():
    from src.graph import _finalize_node

    report = ReportOutput(
        title="T", executive_summary="Summary.", sections=["## A\nBody."]
    )
    state = {
        "report": report,
        "critique": CritiqueOutput(approved=False, score=5, feedback="Still weak."),
    }
    result = _finalize_node(state)
    assert "max revision limit" in result["final_report"].lower()


def test_finalize_node_no_warning_when_approved():
    from src.graph import _finalize_node

    report = ReportOutput(
        title="T", executive_summary="Summary.", sections=["## A\nBody."]
    )
    state = {
        "report": report,
        "critique": CritiqueOutput(approved=True, score=9, feedback=""),
    }
    result = _finalize_node(state)
    assert "max revision limit" not in result["final_report"].lower()