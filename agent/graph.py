"""
agent/graph.py
LangGraph-powered AutoStream conversational agent.

State machine:
  greet ──► route ──► answer    (intent: greeting / product query)
                 └──► qualify   (intent: high-intent lead)
                        └──► capture  (all lead fields collected)
"""

from __future__ import annotations

import os
import re
from typing import Annotated, Literal, TypedDict

import google.generativeai as genai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.rag import retrieve
from tools.lead_capture import mock_lead_capture

# ── LLM ─────────────────────────────────────────────────────────────────────

genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
_MODEL = genai.GenerativeModel("gemini-2.5-flash-lite")


def _call_gemini(prompt: str) -> str:
    """Single entry-point for all Gemini calls."""
    response = _MODEL.generate_content(prompt)
    return response.text.strip()


# ── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Full conversation history (add_messages merges lists automatically)
    messages: Annotated[list, add_messages]

    # Current intent classification
    intent: Literal["greeting", "product_inquiry", "high_intent", "unknown"]

    # Lead fields collected during qualification
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None

    # True once lead_capture tool has fired
    lead_captured: bool

    # Which field to ask for next
    collecting_field: Literal["name", "email", "platform", "done"] | None


# ── Helper: build a flat prompt from system + history + optional new message ──

def _build_prompt(system: str, history: list, user_msg: str | None = None) -> str:
    """
    Converts LangChain message objects + a system string into a single
    text prompt suitable for Gemini's generate_content().
    """
    parts: list[str] = [f"[SYSTEM]\n{system}\n"]

    for msg in history:
        if isinstance(msg, HumanMessage):
            parts.append(f"[USER]\n{msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"[ASSISTANT]\n{msg.content}")
        elif isinstance(msg, SystemMessage):
            parts.append(f"[SYSTEM]\n{msg.content}")

    if user_msg:
        parts.append(f"[USER]\n{user_msg}")

    parts.append("[ASSISTANT]")
    return "\n\n".join(parts)


def _chat(system: str, history: list, user_msg: str | None = None) -> str:
    prompt = _build_prompt(system, history, user_msg)
    return _call_gemini(prompt)


# ── Node: router ─────────────────────────────────────────────────────────────

INTENT_SYSTEM = """You are an intent classifier for AutoStream, a SaaS video-editing platform.
Classify the user's latest message into EXACTLY ONE of these labels:
  greeting        – casual hello / small talk
  product_inquiry – asking about features, pricing, plans, policies, trials
  high_intent     – clearly wants to sign up, try, or purchase a plan

Reply with ONLY the label and nothing else."""

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|good\s+(morning|afternoon|evening))([!,.?\s]+)?$",
    flags=re.IGNORECASE,
)
_HIGH_INTENT_RE = re.compile(
    r"\b(sign\s?up|signup|try|subscribe|get\s+started|buy|purchase|start\s+trial|join)\b",
    flags=re.IGNORECASE,
)
_PRODUCT_RE = re.compile(
    r"\b(pricing|price|plan|plans|feature|features|policy|policies|trial)\b",
    flags=re.IGNORECASE,
)
_DIRECT_HIGH_INTENT_RE = re.compile(
    r"\b(i\s+want\s+(the\s+)?(pro|basic)\s+plan|sign\s*me\s*up|start\s+(the\s+)?(pro|basic)\s+plan|buy\s+(the\s+)?(pro|basic)\s+plan|purchase\s+(the\s+)?(pro|basic)\s+plan|get\s+started)\b",
    flags=re.IGNORECASE,
)
_PLAN_SELECTION_RE = re.compile(
    r"\b(pro\s+plan|basic\s+plan|pricing\s+plan|plan|plans)\b",
    flags=re.IGNORECASE,
)
_NUMERIC_PLAN_RE = re.compile(r"^\s*(1|2)\s*$")
_BASIC_SELECTION_RE = re.compile(r"^\s*(basic|basic\s+plan)\s*$", flags=re.IGNORECASE)
_PRO_SELECTION_RE = re.compile(r"^\s*(pro|pro\s+plan)\s*$", flags=re.IGNORECASE)
_CONTINUE_RE = re.compile(r"^\s*(yes|yeah|yep|ok|okay|sure|start|continue)\s*[!,.?]*\s*$", flags=re.IGNORECASE)
_AFFIRMATIVE_RE = re.compile(r"^\s*(yes|yeah|yep|ok|okay|sure)\s*[!,.?]*\s*$", flags=re.IGNORECASE)
_NEGATIVE_RE = re.compile(r"^\s*(no|nope|nah)\s*[!,.?]*\s*$", flags=re.IGNORECASE)

PRICING_MENU_TEXT = (
    "AutoStream offers two plans:\n\n"
    "1. Basic Plan — $29/month\n"
    "2. Pro Plan — $79/month\n\n"
    'You can:\n'
    '- Type "1" or "Basic" to explore Basic plan\n'
    '- Type "2" or "Pro" to explore Pro plan\n'
    '- Type "start pro" or "start basic" to begin immediately'
)


def _is_new_conversation(state: AgentState) -> bool:
    """True only at the beginning before qualification or capture has started."""
    human_count = sum(1 for m in state["messages"] if isinstance(m, HumanMessage))
    return (
        human_count <= 1
        and state.get("collecting_field") is None
        and not state.get("lead_name")
        and not state.get("lead_email")
        and not state.get("lead_platform")
        and not state.get("lead_captured")
    )


def _is_negative_reply(text: str) -> bool:
    return bool(_NEGATIVE_RE.match(text.strip()))


def _pricing_flow_active(state: AgentState) -> bool:
    """Detect whether the conversation is currently in pricing exploration."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            content = msg.content.lower()
            if "autoStream offers two plans:".lower() in content:
                return True
            if "would you like to continue with the basic plan?" in content:
                return True
            if "would you like to continue with the pro plan?" in content:
                return True
    return False


def _selected_plan(text: str) -> str | None:
    """Map a compact pricing selection to a concrete plan name."""
    cleaned = text.strip().lower()
    if _NUMERIC_PLAN_RE.match(cleaned):
        return "Basic" if cleaned == "1" else "Pro"
    if _BASIC_SELECTION_RE.match(cleaned):
        return "Basic"
    if _PRO_SELECTION_RE.match(cleaned):
        return "Pro"
    return None


def _direct_plan_intent(text: str) -> bool:
    """Detect explicit signup intent that should bypass the option flow."""
    return bool(_DIRECT_HIGH_INTENT_RE.search(text))


def _rule_based_intent(state: AgentState, text: str) -> str | None:
    """Cheap first-pass intent routing with LLM fallback for ambiguous turns."""
    if state.get("collecting_field") in {"name", "email", "platform"} and not state.get("lead_captured"):
        return "high_intent"

    msg = text.strip()
    if not msg:
        return None
    if _is_negative_reply(msg):
        return None
    if _pricing_flow_active(state) and _selected_plan(msg):
        return "product_inquiry"
    if _CONTINUE_RE.match(msg) and _pricing_flow_active(state):
        return "high_intent"
    if _direct_plan_intent(msg):
        return "high_intent"
    if _AFFIRMATIVE_RE.match(msg):
        return "high_intent"
    if _pricing_flow_active(state) and _NUMERIC_PLAN_RE.match(msg):
        return "product_inquiry"
    if _pricing_flow_active(state) and (_BASIC_SELECTION_RE.match(msg) or _PRO_SELECTION_RE.match(msg)):
        return "product_inquiry"
    if _PLAN_SELECTION_RE.search(msg) and not _pricing_flow_active(state):
        return "high_intent"
    if _GREETING_RE.match(msg) and _is_new_conversation(state):
        return "greeting"
    if _HIGH_INTENT_RE.search(msg):
        return "high_intent"
    if _PRODUCT_RE.search(msg):
        return "product_inquiry"
    return None


def router_node(state: AgentState) -> AgentState:
    
    """Classify intent from the latest human message."""
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    # LOCK FLOW DURING LEAD COLLECTION (CRITICAL FIX)
    if state.get("collecting_field") in {"name", "email", "platform"} and not state.get("lead_captured"):
        return {**state, "intent": "high_intent"}

    rule_intent = _rule_based_intent(state, last_human)
    if rule_intent is not None:
        return {**state, "intent": rule_intent}

    raw = _chat(INTENT_SYSTEM, [], last_human).lower().strip()

    if "high_intent" in raw:
        intent = "high_intent"
    elif "product_inquiry" in raw or "product" in raw:
        intent = "product_inquiry"
    elif "greeting" in raw:
        intent = "greeting"
    else:
        purchase_signals = {"sign up", "signup", "try", "subscribe", "get started",
                            "buy", "purchase", "start", "join"}
        if any(sig in last_human.lower() for sig in purchase_signals):
            intent = "high_intent"
        else:
            intent = "product_inquiry"

    return {**state, "intent": intent}


# ── Node: answer (greeting + product queries) ────────────────────────────────

ANSWER_SYSTEM = """You are AutoStream's friendly AI assistant.
AutoStream provides automated AI-powered video editing tools for content creators.

Use ONLY the provided knowledge base context to answer product and pricing questions.
Be concise, helpful, and end with a soft call-to-action when relevant.
If the user seems interested in signing up, encourage them to share their details.

Knowledge Base Context:
{context}
"""


def answer_node(state: AgentState) -> AgentState:
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    if _is_negative_reply(last_human):
        new_msg = AIMessage(
            content="Alright, let me know if you need anything else."
        )
        return {**state, "messages": state["messages"] + [new_msg]}

    if state.get("intent") == "greeting" and _is_new_conversation(state):
        new_msg = AIMessage(
            content="Hi! I can help you with pricing, plans, and features. What would you like to know?"
        )
        return {**state, "messages": state["messages"] + [new_msg]}

    if state.get("intent") == "product_inquiry" and (
        re.search(r"\b(pricing|price|plans?|plan)\b", last_human, flags=re.IGNORECASE)
        and not _pricing_flow_active(state)
    ):
        new_msg = AIMessage(content=PRICING_MENU_TEXT)
        return {**state, "messages": state["messages"] + [new_msg]}

    selected_plan = _selected_plan(last_human) if _pricing_flow_active(state) else None
    if selected_plan:
        context = retrieve(f"{selected_plan} plan")
        system = (
            f"You are AutoStream's sales assistant. The user selected the {selected_plan} plan.\n\n"
            f"Use ONLY the provided knowledge base context to briefly summarize the {selected_plan} plan with clear bullets.\n"
            f"End with exactly: Would you like to continue with the {selected_plan} plan?\n\n"
            f"Knowledge Base Context:\n{context}"
        )
        reply = _chat(system, state["messages"][:-1], last_human)
        new_msg = AIMessage(content=reply)
        return {**state, "messages": state["messages"] + [new_msg]}

    context = retrieve(last_human)
    system = ANSWER_SYSTEM.format(context=context)
    reply = _chat(system, state["messages"][:-1], last_human)
    new_msg = AIMessage(content=reply)
    return {**state, "messages": state["messages"] + [new_msg]}


# ── Node: qualify (collect lead fields one at a time) ────────────────────────

def _extract_field(field: str, text: str) -> str | None:
    """Try to extract a specific field value from free-form user text."""
    text = text.strip()
    if field == "email":
        match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
        return match.group(0) if match else None
    if field == "platform":
        platforms = ["youtube", "instagram", "tiktok", "twitch", "facebook",
                     "snapchat", "linkedin", "twitter", "x"]
        tl = text.lower()
        for p in platforms:
            if p in tl:
                return p.capitalize()
        if len(text.split()) <= 3:
            return text
        return None
    if field == "name":
        # Prefer explicit self-introduction patterns.
        match = re.search(r"^(?:my name is|i am|i'm|this is)\s+(.+)$", text, flags=re.IGNORECASE)
        candidate = match.group(1).strip() if match else text
        candidate = re.sub(r"^[\s\-,:;.!?]+|[\s\-,:;.!?]+$", "", candidate)

        if not candidate:
            return None

        tl = candidate.lower()

        # Reject obvious non-name phrases (pricing intent, confirmations, platform mentions, etc.)
        if re.search(
            r"\b(plan|pricing|price|pro|basic|trial|subscribe|buy|yes|yeah|yep|ok|okay|sure|no|nope|nah|youtube|instagram|tiktok|twitch|facebook|snapchat|linkedin|twitter|x)\b",
            tl,
        ):
            return None

        # Basic validation: person-like tokens only.
        if "@" in candidate or re.search(r"\d", candidate):
            return None

        tokens = candidate.split()
        if 1 <= len(tokens) <= 4 and all(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", tok) for tok in tokens):
            return candidate
    return None


def _next_missing_field(name: str | None, email: str | None, platform: str | None) -> str:
    """Return the next required lead field in strict order."""
    if not name:
        return "name"
    if not email:
        return "email"
    if not platform:
        return "platform"
    return "done"


def _qualify_reply(name: str | None, next_field: str) -> str:
    if next_field == "name":
        return "Great choice! I'll help you get started. What's your full name?"
    if next_field == "email":
        first_name = (name or "there").split()[0]
        return f"Thanks, {first_name}! What's your email address?"
    if next_field == "platform":
        return "Perfect. Which platform do you create on? (YouTube, Instagram, etc.)"
    return ""


def qualify_node(state: AgentState) -> AgentState:
    """Collect lead fields step by step."""
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    name     = state.get("lead_name")
    email    = state.get("lead_email")
    platform = state.get("lead_platform")

    # Enforce strict collection order: name -> email -> platform.
    next_field = _next_missing_field(name, email, platform)

    if _is_negative_reply(last_human):
        new_messages = state["messages"] + [AIMessage(content="Alright, let me know if you need anything else.")]
        return {
            **state,
            "messages": new_messages,
            "lead_name": name,
            "lead_email": email,
            "lead_platform": platform,
            "collecting_field": next_field,
        }

    if next_field == "name":
        extracted = _extract_field("name", last_human)
        if extracted:
            name = extracted
    elif next_field == "email":
        extracted = _extract_field("email", last_human)
        if extracted:
            email = extracted
    elif next_field == "platform":
        extracted = _extract_field("platform", last_human)
        if extracted:
            platform = extracted

    next_field = _next_missing_field(name, email, platform)

    new_messages = state["messages"]
    if next_field != "done":
        reply = _qualify_reply(name, next_field)
        new_messages = state["messages"] + [AIMessage(content=reply)]

    return {
        **state,
        "messages": new_messages,
        "lead_name": name,
        "lead_email": email,
        "lead_platform": platform,
        "collecting_field": next_field,
    }


# ── Node: capture (fire the tool) ────────────────────────────────────────────

def capture_node(state: AgentState) -> AgentState:
    """Fire mock_lead_capture and confirm to the user."""
    result = mock_lead_capture(
        name=state["lead_name"],
        email=state["lead_email"],
        platform=state["lead_platform"],
    )
    lead_id = result["lead_id"]

    confirmation = (
        f" You're all set, {state['lead_name']}! "
        f"We've captured your details and your account is being set up.\n\n"
        f"**Confirmation ID:** `{lead_id}`\n\n"
        f"You'll receive a welcome email at **{state['lead_email']}** shortly. "
        f"Our team will reach out to help you get started with AutoStream on {state['lead_platform']}. "
        f"Feel free to ask if you have any other questions! "
    )

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=confirmation)],
        "lead_captured": True,
        "collecting_field": "done",
    }


# ── Routing edges ─────────────────────────────────────────────────────────────

def route_after_router(state: AgentState) -> str:
    if state.get("lead_captured"):
        return "answer"
    if state["intent"] == "high_intent":
        return "qualify"
    return "answer"


def route_after_qualify(state: AgentState) -> str:
    if state.get("collecting_field") == "done":
        return "capture"
    return END


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router",  router_node)
    g.add_node("answer",  answer_node)
    g.add_node("qualify", qualify_node)
    g.add_node("capture", capture_node)

    g.add_edge(START, "router")

    g.add_conditional_edges(
        "router",
        route_after_router,
        {"answer": "answer", "qualify": "qualify"},
    )

    g.add_edge("answer", END)

    g.add_conditional_edges(
        "qualify",
        route_after_qualify,
        {"capture": "capture", END: END},
    )

    g.add_edge("capture", END)

    return g.compile()


# Singleton
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public interface ──────────────────────────────────────────────────────────

def initial_state() -> AgentState:
    return AgentState(
        messages=[],
        intent="unknown",
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        lead_captured=False,
        collecting_field=None,
    )


def chat(state: AgentState, user_input: str) -> tuple[AgentState, str]:
    """
    Process one user turn.
    Returns (new_state, agent_reply_text).
    """
    graph = get_graph()
    new_state = graph.invoke(
        {**state, "messages": state["messages"] + [HumanMessage(content=user_input)]}
    )
    last_ai = next(
        (m.content for m in reversed(new_state["messages"]) if isinstance(m, AIMessage)),
        "Sorry, I didn't understand that.",
    )
    return new_state, last_ai
