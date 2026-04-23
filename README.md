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

▶️ [Watch Demo Video](https://drive.google.com/file/d/1mMOsLMKN9MTO_JhNCMdeJpa7Q25qNRvN/view?usp=sharing)

---

## Features

* **Intent Detection**

  * Classifies user input into:

    * Greeting
    * Product inquiry
    * High-intent lead

* **RAG-Based Responses**

  * Answers questions using a local knowledge base (pricing, features, policies)

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
We offer two plans:
- Basic Plan ($29/month)
- Pro Plan ($79/month)

User: I want to try the Pro plan

Assistant:
Great choice! Could you tell me your full name?

User: Full Name

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
