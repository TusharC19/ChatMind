import os
import re
import time

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


load_dotenv()


class ExtractionResult(BaseModel):
    message_id: int = Field(
        description="The message_id of the candidate conversation being analyzed."
    )

    is_important: bool = Field(
        description="Whether this conversation contains useful actionable information."
    )

    type: str = Field(
        description="One of: task, request, deadline, decision, information, none."
    )

    task: str | None = Field(
        default=None,
        description="The action that needs to be performed, if any."
    )

    deadline: str | None = Field(
        default=None,
        description="Deadline or relevant time mentioned in the conversation, if any."
    )

    priority: str = Field(
        description="One of: low, medium, high."
    )

    confidence: float = Field(
        description="Confidence from 0.0 to 1.0."
    )

    evidence: str | None = Field(
        default=None,
        description=(
            "The exact message text or short phrase from the "
            "conversation that supports the classification. "
            "Use null when there is no sufficient evidence."
        )
    )


class BatchExtractionResult(BaseModel):
    results: list[ExtractionResult]


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in .env"
    )


client = genai.Client(api_key=api_key)


def extract_information(context):
    """
    Analyze a single conversation context using Gemini
    and return structured information.
    """

    previous_messages = context["previous"]
    current_message = context["current"]
    next_messages = context["next"]

    previous_text = "\n".join(
        f'{msg["user"]}: {msg["message"]}'
        for msg in previous_messages
    )

    next_text = "\n".join(
        f'{msg["user"]}: {msg["message"]}'
        for msg in next_messages
    )

    current_text = (
        f'{current_message["user"]}: '
        f'{current_message["message"]}'
    )

    message_id = current_message["message_id"]

    prompt = f"""
You are a conversation intelligence system analyzing a
WhatsApp conversation.

Your job is to identify actionable or important information.

Understand informal language, Hinglish, abbreviations,
and casual WhatsApp-style conversation.

Do NOT invent information that is not present.

Use the surrounding messages to understand the current message.

MESSAGE_ID:
{message_id}

PREVIOUS MESSAGES:
{previous_text}

CURRENT MESSAGE:
{current_text}

NEXT MESSAGES:
{next_text}

Classify the conversation into one of:

- task
- request
- deadline
- decision
- information
- none

TASK vs REQUEST:

A TASK means the speaker or the speaker's side is expected
to perform the action.

Examples:
- "I need to submit the assignment." → task
- "I'll send the code tonight." → task
- "I will complete the form." → task

A REQUEST means the speaker is asking ANOTHER PERSON
to perform an action or provide something.

Examples:
- "Can you send me the assignment?" → request
- "Please send the PYQ solution." → request
- "Mujhe notes bhej do." → request

Do NOT classify a message as a REQUEST merely because it
contains an action verb such as send, solve, submit, check,
prepare, or share.

Determine WHO is expected to perform the action using the
current message and surrounding context.

If the speaker describes their own responsibility,
intention, or planned action, classify it as TASK.

If the speaker asks another person to act or provide
something, classify it as REQUEST.

If the context does not establish who should perform the
action, do not invent an assignee. Prefer a conservative
classification with lower confidence.

DEADLINE:

A DEADLINE is used when a date, day, time, or time
constraint is explicitly mentioned AND is clearly associated
with an action, requirement, or commitment.

Examples:
- "Submit the assignment tomorrow." → deadline
- "Kal submit karna hai." → deadline

Do not classify every message containing a time as a
deadline. The time must be meaningful to an action,
requirement, or commitment.

INFORMATION:

An INFORMATION message contains a useful fact, update,
announcement, or knowledge that does not necessarily
require an action.

DECISION:

A DECISION means the conversation clearly indicates that
something has been finalized, selected, agreed upon,
or decided.

Examples:
- "We'll use React." → decision
- "Topic finalized: Face Recognition." → decision
- "Let's go with option 2." → decision

Do not classify a suggestion as a decision unless the
conversation indicates that it was actually accepted
or finalized.

CONTEXT INTERPRETATION:

The CURRENT MESSAGE is the primary message being classified.

PREVIOUS and NEXT messages are provided to clarify its meaning.

Use context to determine:
- who is speaking
- who is being addressed
- who is expected to perform an action
- what an action refers to
- what a deadline applies to
- whether an action is personal or requested from another person

Context may clarify the meaning, but NEVER invent an action,
person, deadline, responsibility, or decision that is not
supported by the conversation.

AMBIGUOUS ACTIONS:

If a message could reasonably be interpreted as either a
TASK or REQUEST and the available context does not resolve
the ambiguity, do not force a classification.

Prefer NONE or INFORMATION with lower confidence rather
than making an unsupported assumption.

If there is an actionable task, describe it briefly.

If a deadline or time is mentioned and clearly applies to
an action, extract it. Otherwise use null.

Use:
- high priority for urgent or deadline-sensitive items
- medium priority for useful requests or tasks
- low priority for less important information

If the context is insufficient, do not guess.

STRICT ACCURACY RULES:

1. DO NOT INVENT INFORMATION.
   Only use information explicitly present in the
   conversation or clearly established by its surrounding
   messages.

2. DO NOT ASSUME WHAT A VAGUE MESSAGE REFERS TO.

3. The TASK field must describe only an action that is
   actually supported by the conversation.

4. The DEADLINE field must contain only a time/date
   explicitly supported by the conversation.

5. is_important should be TRUE only when the conversation
   contains genuinely useful information, an actionable
   request/task, a meaningful deadline, or a clear decision.

6. If the conversation is casual, vague, or insufficient,
   use:
   - type = none
   - task = null
   - deadline = null
   - evidence = null

EVIDENCE REQUIREMENT:

For every result that is not NONE, provide an "evidence"
field containing the exact original message text or a short
exact phrase from the conversation that directly supports
your classification.

Do NOT write an explanation in the evidence field.

Evidence must come directly from the conversation.

For tasks, the evidence must support the actual action
described in the task.

For requests, the evidence must support that another person
is being asked to perform or provide something.

For deadlines, the evidence must support both the existence
of the deadline and what it applies to.

For decisions, the evidence must support that the decision
was actually finalized or agreed upon.

Do not use surrounding context to invent an action.
Context may clarify a message, but the resulting task,
request, deadline, or decision must still be directly
supported by the conversation.

Return the MESSAGE_ID exactly as provided.

Return only the requested structured output.
"""

    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ExtractionResult,
                },
            )

            return ExtractionResult.model_validate_json(
                response.text
            )

        except Exception as error:

            error_text = str(error)

            # Handle Gemini rate-limit errors.
            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:

                retry_match = re.search(
                    r"retryDelay.*?(\d+)",
                    error_text
                )

                if retry_match:
                    wait_time = int(retry_match.group(1)) + 1
                else:
                    wait_time = 60

                if attempt == max_retries - 1:
                    raise RuntimeError(
                        "Gemini rate limit exceeded. "
                        f"Please wait before trying again: {error}"
                    ) from error

                print(
                    f"Gemini rate limit reached. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)
                continue

            # Other temporary errors.
            if attempt == max_retries - 1:
                raise RuntimeError(
                    "Gemini request failed after "
                    f"{max_retries} attempts: {error}"
                ) from error

            wait_time = 2 ** attempt

            print(
                f"Gemini request failed. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


def extract_batch_information(batch):
    """
    Analyze multiple candidate conversations
    using a single Gemini request.
    """

    conversation_blocks = []

    for item in batch:

        message_id = item["message_id"]
        context = item["context"]

        previous_text = "\n".join(
            f'{msg["user"]}: {msg["message"]}'
            for msg in context["previous"]
        )

        current = context["current"]

        current_text = (
            f'{current["user"]}: '
            f'{current["message"]}'
        )

        next_text = "\n".join(
            f'{msg["user"]}: {msg["message"]}'
            for msg in context["next"]
        )

        conversation = f"""
MESSAGE_ID: {message_id}

PREVIOUS MESSAGES:
{previous_text}

CURRENT MESSAGE:
{current_text}

NEXT MESSAGES:
{next_text}
"""

        conversation_blocks.append(conversation)

    all_conversations = "\n\n--------------------\n\n".join(
        conversation_blocks
    )

    prompt = f"""
You are a conversation intelligence system analyzing
multiple WhatsApp conversations.

Analyze EACH conversation independently.

Your goal is to identify genuinely useful information
from the conversation, especially tasks, requests,
deadlines, decisions, and important updates.

Understand:
- informal WhatsApp language
- Hinglish
- abbreviations
- spelling mistakes
- casual conversation

TASK vs REQUEST — IMPORTANT DISTINCTION:

A TASK means the speaker or the speaker's side is expected
to perform the action.

Examples:
- "I need to submit the assignment." → task
- "I'll send the code tonight." → task
- "I will complete the form." → task

A REQUEST means the speaker is asking ANOTHER PERSON
to perform an action or provide something.

Examples:
- "Can you send me the assignment?" → request
- "Please send the PYQ solution." → request
- "Mujhe notes bhej do." → request

DO NOT classify a message as a REQUEST merely because it
contains an imperative/action verb such as:
send, solve, submit, check, prepare, or share.

Determine WHO is expected to perform the action using the
current message and surrounding context.

If the speaker describes their own responsibility,
intention, or planned action, classify it as TASK.

If the speaker asks another person to act or provide
something, classify it as REQUEST.

If the context does not establish who should perform the
action, do not invent an assignee. Prefer a conservative
classification with lower confidence.

IMPORTANT:
Do not assume that every imperative sentence is a request.
The expected actor matters.

DEADLINE:

A DEADLINE is used when a specific date, day, time, or
time constraint is explicitly mentioned AND is clearly
associated with an action, requirement, or commitment.

Examples:
- "Submit the assignment tomorrow." → deadline
- "Kal submit karna hai." → deadline

Do not classify every message containing a time as a
deadline. The time must be meaningful to an action,
requirement, or commitment.

INFORMATION:

An INFORMATION message contains a useful fact, update,
announcement, or knowledge that does not necessarily
require an action.

DECISION:

A DECISION means the conversation clearly indicates that
something has been finalized, selected, agreed upon,
or decided.

Examples:
- "We'll use React." → decision
- "Topic finalized: Face Recognition." → decision
- "Let's go with option 2." → decision

Do not classify a suggestion as a decision unless the
conversation indicates that it was actually accepted
or finalized.

CONTEXT INTERPRETATION:

The CURRENT MESSAGE is the primary message being classified.

PREVIOUS and NEXT messages are provided to clarify its meaning.

Use context to determine:
- who is speaking
- who is being addressed
- who is expected to perform an action
- what an action refers to
- what a deadline applies to
- whether an action is personal or requested from another person
- whether a decision was actually finalized

Context may clarify the meaning, but NEVER invent an action,
person, deadline, responsibility, or decision that is not
supported by the conversation.

AMBIGUOUS ACTIONS:

If a message could reasonably be interpreted as either
a TASK or REQUEST and the available context does not
resolve the ambiguity, do not force a classification.

Prefer NONE or INFORMATION with lower confidence rather
than making an unsupported assumption.

STRICT ACCURACY RULES:

1. DO NOT INVENT INFORMATION.
   Only use information explicitly present in the
   conversation or clearly established by its surrounding
   messages.

2. DO NOT ASSUME WHAT A VAGUE MESSAGE REFERS TO.
   For example, if someone only says "kal raat ko",
   do not invent what they will do unless the surrounding
   conversation clearly establishes the action.

3. A TASK means the speaker or the speaker's side is
   expected to perform a clear action.

4. A REQUEST means the speaker is asking another person
   to perform an action or provide something.

5. A DEADLINE means a specific date, day, time, or
   time constraint is explicitly mentioned AND the
   surrounding conversation clearly establishes what
   it applies to.

6. An INFORMATION message contains a useful fact,
   update, announcement, or knowledge that does not
   necessarily require an action.

7. A DECISION means the conversation clearly indicates
   that something has been decided, finalized, selected,
   or agreed upon.

8. Use NONE when:
   - the message is casual conversation
   - the context is insufficient
   - the message is too vague to classify reliably
   - there is no genuinely useful information

9. If you are uncertain, prefer NONE or INFORMATION
   rather than making an unsupported assumption.

10. The TASK field must describe only an action that is
    actually supported by the conversation.

11. The DEADLINE field must contain only a time/date
    explicitly supported by the conversation.
    Do not create a deadline from assumptions.

12. is_important should be TRUE only when the
    conversation contains genuinely useful information,
    an actionable request/task, a meaningful deadline,
    or a clear decision.

13. Preserve the MESSAGE_ID exactly as provided.

14. CONTEXT MUST NOT BE USED TO INVENT FACTS.
    Context may clarify the meaning of the current message,
    but every extracted result must remain supported by
    the conversation.

15. EVIDENCE REQUIREMENT:
    For every result that is not NONE, provide an
    "evidence" field containing the exact original
    message text or a short exact phrase from the
    conversation that directly supports your decision.

16. Do NOT write an explanation in the evidence field.
    Evidence must come directly from the conversation.

17. If there is no clear supporting evidence, use:
    - type = none
    - task = null
    - deadline = null
    - evidence = null

18. For tasks, the evidence must support the actual action
    described in the task.

19. For requests, the evidence must support that another
    person is being asked to perform or provide something.

20. For deadlines, the evidence must support both the
    existence of the deadline and what it applies to.

21. For decisions, the evidence must support that the
    decision was actually finalized or agreed upon.

22. Do not use surrounding context to invent an action.
    Context may clarify a message, but the resulting task,
    request, deadline, or decision must still be directly
    supported by the conversation.

23. If a TASK and REQUEST interpretation are both possible
    and context does not resolve the difference, prefer
    NONE or INFORMATION with lower confidence rather than
    inventing who is responsible.

CONVERSATIONS:

{all_conversations}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": BatchExtractionResult,
        },
    )

    return BatchExtractionResult.model_validate_json(
        response.text
    )