# AutoStream – Social-to-Lead Agentic Workflow

## Overview

This project implements a conversational AI agent for a fictional SaaS product, **AutoStream**, an AI-powered video editing platform.

The agent is designed to:

* Understand user intent
* Answer product-related queries using a knowledge base (RAG)
* Identify high-intent users
* Collect lead information
* Trigger backend actions (lead capture)

---

## Demo Video

▶️ [Watch Demo Video](https://drive.google.com/file/d/1RNI2EZFm9lYxPG8osTSOC1NMRKv49tyd/view?usp=sharing)

---

## Features

* **Intent Detection**

  * Classifies user input into:

    * Greeting
    * Product inquiry
    * High-intent lead

* **RAG-Based Responses**

  * Answers questions using a local knowledge base (pricing, features, policies)

* **Structured Option-Based Interaction**

  * Guides users with clear choices (Basic / Pro or numeric selection)
  * Supports “tell me more” style exploration before conversion

* **Multi-Turn Conversation**

  * Maintains state across multiple interactions

* **Lead Qualification Flow**

  * Collects user details in order:

    * Name → Email → Platform

* **Tool Execution**

  * Calls `mock_lead_capture()` only after all required fields are collected

---

## Architecture

This agent is built using **LangGraph**, enabling a structured, state-driven workflow.

* **State Management**

  * Maintains conversation history and lead data (name, email, platform)
  * Ensures continuity across multiple turns

* **Intent Routing**

  * Uses a hybrid approach:

    * Rule-based detection for common inputs
    * LLM fallback for ambiguous cases

* **RAG (Retrieval-Augmented Generation)**

  * Retrieves relevant information from a local JSON knowledge base
  * Injects context into responses for accurate answers

* **Lead Qualification Logic**

  * Deterministic flow ensures correct order of data collection
  * Prevents missing or misordered inputs

* **Tool Execution Control**

  * Lead capture is triggered only after all fields are collected
  * Prevents premature API calls

* **UX Design Improvement**

  * Uses structured prompts to guide users through plan selection and exploration before lead conversion

---

## Tech Stack

* Python 3.9+
* LangGraph / LangChain
* Google Gemini 2.5 Flash Lite
* Local JSON Knowledge Base

---

## Project Structure

```
autostream-agent/
│
├── main.py
├── test_agent.py
├── requirements.txt
├── .env.example
│
├── agent/
│   ├── graph.py
│   └── rag.py
│
├── tools/
│   └── lead_capture.py
│
└── knowledge_base/
    └── autostream_kb.json
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

### 3. Run the agent

```bash
python main.py
```

---

## Example Conversation

```
User: Tell me about pricing

Assistant:
AutoStream offers two plans:

1. Basic Plan — $29/month
2. Pro Plan — $79/month

You can:
- Type "1" or "Basic" to explore Basic plan
- Type "2" or "Pro" to explore Pro plan
- Type "start pro" to begin immediately

User: 2

Assistant:
The Pro Plan includes:
- Unlimited videos
- 4K export
- AI captions and subtitles
- Advanced editing tools
- Priority rendering
- 24/7 support
- Custom branding
- Analytics dashboard

Would you like to continue with the Pro plan?

User: yes

Assistant:
Great! What’s your full name?

User: Name

Assistant:
Thanks Name! What's your email address?

User: example@email.com

Assistant:
Which platform do you create on?

User: YouTube

Assistant:
Lead captured successfully.
```

---

## WhatsApp Integration (Approach)

To deploy this agent on WhatsApp:

* Use WhatsApp Business API (e.g., via Twilio)
* Receive incoming messages through webhooks
* Process messages using the agent backend
* Maintain user session state (e.g., Redis or database)
* Send responses back via the API

---

## Notes

This implementation focuses on:

* Correct agent workflow
* Clean state management
* Controlled tool execution
* Efficient API usage

The system is designed to be simple, modular, and aligned with real-world agent architectures.