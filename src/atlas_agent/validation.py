"""Milestone 3: implement three selected validation responsibilities."""

from __future__ import annotations

from .agent_models import ToolCallDecision
from .models import AgentResponse, LabRequest, TraceEvent
from .policy_context import ReservationContext, build_reservation_context


def validate_request_binding(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
) -> list[str]:
    errors: list[str] = []
    args = decision.arguments
    if (
        decision.tool in {"get_user_authorizations", "reserve_resource"}
        or "user_id" in args
    ) and args.get("user_id") != request.authenticated_user_id:
        errors.append("user_id_must_match_authenticated_user")
    for field in ("project_id", "resource_id", "start", "end"):
        expected = getattr(request, field)
        if expected is None:
            if decision.tool == "reserve_resource":
                errors.append(f"{field}_unresolved_in_request")
        elif field in args and args[field] != expected:
            errors.append(f"{field}_does_not_match_request")
    return errors


def validate_reservation_preconditions(context: ReservationContext) -> list[str]:
    requirements = {
        "arguments_complete": "reservation_arguments_incomplete",
        "policy_retrieved": "policy_not_retrieved",
        "authorizations_checked": "authorizations_not_checked",
        "resource_status_checked": "resource_status_not_checked",
        "project_authorized": "project_not_authorized",
        "qualification_current": "required_qualification_missing",
        "resource_available": "resource_not_available",
        "calibration_current": "calibration_not_current",
        "maintenance_clear": "resource_under_maintenance",
        "after_hours_approval_satisfied": "after_hours_approval_required",
    }
    return [error for fact, error in requirements.items() if not getattr(context, fact)]


def validate_proposed_tool_call(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
    trace: list[TraceEvent],
) -> list[str]:
    """Supplied wrapper. Do not modify this function."""
    errors = validate_request_binding(decision, request=request)
    if decision.tool == "reserve_resource":
        context = build_reservation_context(decision, trace)
        errors.extend(validate_reservation_preconditions(context))
    return errors


def retrieved_passage_ids(trace: list[TraceEvent]) -> set[str]:
    """Supplied helper. Do not modify this function."""
    identifiers: set[str] = set()
    for event in trace:
        if event.observation.tool != "search_lab_documents" or not event.observation.ok:
            continue
        identifiers.update(
            passage["passage_id"]
            for passage in event.observation.data.get("passages", [])
        )
    return identifiers


def validate_final_response(
    response: AgentResponse, trace: list[TraceEvent]
) -> list[str]:
    errors: list[str] = []
    missing_citations = set(response.citations) - retrieved_passage_ids(trace)
    if missing_citations:
        errors.append("citations_not_retrieved: " + ", ".join(sorted(missing_citations)))
    successful_actions = [
        event.observation.tool
        for event in trace
        if event.mutating and event.observation.ok
    ]
    remaining_actions = successful_actions.copy()
    for action in response.completed_actions:
        if action not in remaining_actions:
            errors.append("completed_action_not_in_successful_trace")
        else:
            remaining_actions.remove(action)
    if response.status == "completed":
        if not successful_actions:
            errors.append("completed_without_successful_mutation")
        if not response.completed_actions:
            errors.append("completed_without_completed_actions")
    else:
        if response.completed_actions:
            errors.append("non_completed_response_claims_completed_actions")
        if successful_actions:
            errors.append("successful_mutation_not_reported_as_completed")
    return errors
