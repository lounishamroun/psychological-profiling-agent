"""
Agent functions: Inspector, Suspect, Profiler, Final Report.

Each function takes the current state, builds a prompt, calls the LLM,
and returns a dict of state updates. LangGraph handles merging these
updates back into the state.
"""

import json
from langfuse import observe
from src.utils import call_llm, parse_json_response, format_conversation
from src.prompts import (
    INSPECTOR_PROMPT,
    SUSPECT_PROMPT,
    PROFILER_PROMPT,
    FINAL_REPORT_PROMPT,
    JUDGE_PROMPT,
)
from src.state import InterrogationState


@observe(name="inspector_agent")
def inspector_agent(state: InterrogationState) -> dict:
    """Generate one strategic question from the Inspector."""
    prompt = INSPECTOR_PROMPT.format(
        suspect_name=state["suspect_profile"].get("name", "the suspect"),
        case_data=json.dumps(state["case_data"], indent=2),
        retrieved_context="\n---\n".join(state.get("retrieved_context", [])),
        conversation_history=format_conversation(state.get("conversation_history", [])),
        profiler_output=json.dumps(state.get("profiler_output", {}), indent=2),
    )
    question = call_llm(prompt, temperature=0.7).strip()

    # Return state updates:
    # - last_question: overwritten (for the suspect to see)
    # - conversation_history: appended (Annotated[list, operator.add])
    return {
        "last_question": question,
        "conversation_history": [{"role": "inspector", "content": question}],
    }


@observe(name="suspect_agent")
def suspect_agent(state: InterrogationState) -> dict:
    """Generate an in-character answer from the Suspect."""
    prompt = SUSPECT_PROMPT.format(
        suspect_name=state["suspect_profile"].get("name", "the suspect"),
        suspect_profile=json.dumps(state["suspect_profile"], indent=2),
        case_data=json.dumps(state["case_data"], indent=2),
        conversation_history=format_conversation(state.get("conversation_history", [])),
        last_question=state["last_question"],
    )
    answer = call_llm(prompt, temperature=1.0).strip()

    return {
        "last_answer": answer,
        "conversation_history": [{"role": "suspect", "content": answer}],
    }


@observe(name="profiler_agent")
def profiler_agent(state: InterrogationState) -> dict:
    """Analyze the suspect's answer and return structured metrics."""
    profiler_ctx = state.get("profiler_context", [])
    profiler_context_str = "\n---\n".join(profiler_ctx) if profiler_ctx else "Aucun exemple disponible"
    prompt = PROFILER_PROMPT.format(
        case_data=json.dumps(state["case_data"], indent=2),
        conversation_history=format_conversation(state.get("conversation_history", [])),
        last_question=state["last_question"],
        last_answer=state["last_answer"],
        profiler_context=profiler_context_str,
    )
    raw = call_llm(prompt, temperature=0.2)
    profiler_output = parse_json_response(raw)

    return {
        "profiler_output": profiler_output,
        "profiler_history": [profiler_output],  # appended via operator.add
        "turn": state["turn"] + 1,              # increment turn counter
    }


@observe(name="final_report_agent")
def final_report_agent(state: InterrogationState) -> dict:
    """Generate the final interrogation assessment report."""
    prompt = FINAL_REPORT_PROMPT.format(
        case_data=json.dumps(state["case_data"], indent=2),
        conversation_history=format_conversation(state.get("conversation_history", [])),
        all_profiler_outputs=json.dumps(state.get("profiler_history", []), indent=2),
    )
    report = call_llm(prompt, temperature=0.3).strip()

    return {"final_report": report}


@observe(name="judge_agent")
def judge_agent(state: InterrogationState) -> dict:
    """Evaluate the entire simulation quality (called after the graph finishes)."""
    prompt = JUDGE_PROMPT.format(
        case_data=json.dumps(state["case_data"], indent=2),
        suspect_profile=json.dumps(state["suspect_profile"], indent=2),
        conversation_history=format_conversation(state.get("conversation_history", [])),
        all_profiler_outputs=json.dumps(state.get("profiler_history", []), indent=2),
        final_report=state.get("final_report", ""),
    )
    raw = call_llm(prompt, temperature=0.0)
    return parse_json_response(raw)
