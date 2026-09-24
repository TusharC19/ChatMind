 # ChatMind --- AI-Powered Conversation Intelligence

ChatMind is a **WhatsApp conversation intelligence platform** that
turns unstructured group conversations into useful, actionable
information.

Instead of only showing chat statistics, ChatMind identifies
**tasks, requests, deadlines, and decisions** hidden inside everyday
conversations and lets users search their conversations using natural
language.

------------------------------------------------------------------------

## 🚀 Why ChatMind?

Important information in WhatsApp groups is often buried among hundreds
or thousands of casual messages.

For example:

> "Please submit the assignment by tomorrow."

A normal chat analytics application may count this as just another
message.

ChatMind can identify that it contains a:

-   **Deadline:** tomorrow
-   **Task:** submit assignment
-   **Priority:** high
-   **Evidence:** "submit the assignment by tomorrow"

This turns a conversation into **actionable information**.

------------------------------------------------------------------------

## ✨ Features

### 📊 Chat Analytics

The original chat-analysis functionality is preserved.

-   Total messages
-   Total words
-   Media shared
-   Links shared
-   Monthly activity timeline
-   Daily activity timeline
-   Most active day
-   Most active month
-   Weekly activity heatmap
-   Most active users
-   Word cloud
-   Most common words
-   Emoji analysis

------------------------------------------------------------------------

### 🧠 AI Conversation Intelligence

The AI pipeline analyzes relevant messages and extracts structured
information such as:

-   **Tasks**
-   **Requests**
-   **Deadlines**
-   **Decisions**
-   Priority
-   Confidence score
-   Evidence from the original message

The system first filters potentially useful messages instead of sending
the entire conversation to the LLM.

------------------------------------------------------------------------

### 🎯 Action Dashboard

The extracted information is organized into a dashboard with filters
for:

-   Type
    -   Tasks
    -   Requests
    -   Deadlines
    -   Decisions
-   Priority
    -   High
    -   Medium
    -   Low

The dashboard also removes duplicate actions before displaying them.

------------------------------------------------------------------------

### 🔎 Evidence & Source Traceability

Every extracted action can be traced back to its original WhatsApp
message.

The source viewer displays:

-   AI evidence
-   Original message
-   Surrounding conversation context
-   Message ID

This makes AI-generated insights **verifiable rather than opaque**.

Example:

``` text
Extracted:
Submit assignment

Deadline:
Tomorrow

Evidence:
"submit the assignment by tomorrow"

Source:
Original WhatsApp conversation
```

------------------------------------------------------------------------

### 💬 Ask Your Conversation

Users can search the chat using natural language.

Example:

``` text
When do I need to submit the assignment?
```

The system searches the conversation semantically and combines:

-   Semantic similarity
-   Keyword matching
-   AI-extracted information

The result can include the relevant message, AI insight, evidence,
priority, confidence, and surrounding context.

------------------------------------------------------------------------

## 🏗️ System Architecture

``` text
                 WhatsApp TXT Export
                         │
                         ▼
                ┌─────────────────┐
                │  Preprocessing  │
                │   + Parsing     │
                └────────┬────────┘
                         │
                         ▼
                  Pandas DataFrame
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Chat Analytics          Candidate Filtering
                                     │
                                     ▼
                              Context Extraction
                                     │
                                     ▼
                              Gemini AI Extraction
                                     │
                                     ▼
                           Validation + Evidence
                                     │
                                     ▼
                              AI Results DataFrame
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
             Action Dashboard                  Semantic Search
                    │                                 │
                    └────────────────┬────────────────┘
                                     ▼
                              Streamlit Interface
```

------------------------------------------------------------------------

## 🔄 AI Processing Pipeline

The AI system is designed as a multi-stage pipeline rather than sending
every message directly to an LLM.

### 1. Preprocessing

The WhatsApp `.txt` export is parsed into structured records containing
information such as:

-   Message ID
-   Date/time
-   User
-   Message
-   Date components
-   Activity periods

### 2. Candidate Filtering

Rule-based patterns identify messages that are potentially actionable.

Examples include:

-   Action words
-   Requests
-   Deadline expressions
-   Decision-related phrases
-   Questions

This reduces unnecessary LLM calls.

For example, in one tested conversation:

``` text
A large WhatsApp conversation
A filtered subset of potentially relevant messages
```

### 3. Context Extraction

Each candidate is processed together with nearby messages.

The default context window is:

``` text
2 previous messages
+
current message
+
2 next messages
```

This provides conversational context for ambiguous messages.

### 4. Gemini Extraction

Gemini converts the conversational context into structured information.

The extraction schema includes:

``` text
message_id
is_important
type
task
deadline
priority
confidence
evidence
```

### 5. Validation

The extracted results are validated before being displayed.

Validation checks include:

-   Valid message IDs
-   Supported result types
-   Valid priorities
-   Confidence values
-   Evidence consistency
-   Evidence presence in the surrounding conversation

Unsupported or insufficiently grounded results can be demoted instead of
being blindly displayed.

### 6. Deduplication

Similar actions are normalized and duplicate results are removed.

The system preserves the highest-confidence occurrence so that its
original source and evidence remain available.

### 7. Semantic Search

Conversation messages are embedded using:

``` text
all-MiniLM-L6-v2
```

and indexed with FAISS.

Search ranking combines:

``` text
60% Semantic Similarity
25% Keyword Matching
15% AI Relevance
```

This allows queries such as:

``` text
When do I need to submit the assignment?
```

to retrieve relevant messages even when the query wording does not
exactly match the original conversation.

------------------------------------------------------------------------

## 🛠️ Tech Stack

### Frontend / Interface

-   Streamlit

### Programming

-   Python

### Data Processing

-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn

### AI / NLP

-   Google Gemini API
-   `google-genai`
-   Sentence Transformers

### Semantic Search

-   FAISS
-   `all-MiniLM-L6-v2`

### Validation

-   Pydantic

### Configuration

-   Python environment variables / `.env`

------------------------------------------------------------------------

## 📁 Project Structure

``` text
ChatMind/
│
├── app.py
├── preprocessor.py
├── helper.py
│
├── ai/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── candidate_filter.py
│   ├── context.py
│   ├── extractor.py
│   ├── pipeline.py
│   ├── postprocessor.py
│   └── semantic_search.py
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

## ⚙️ Installation

### 1. Clone the repository

``` bash
git clone https://github.com/TusharC19/ChatMind.git
cd ChatMind
```

### 2. Create a virtual environment

``` bash
python -m venv venv
```

Activate it on Windows:

``` bash
venv\Scripts\activate
```

On Linux/macOS:

``` bash
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

If dependencies are not already listed in `requirements.txt`, the core
AI/search packages include:

``` bash
pip install google-genai sentence-transformers faiss-cpu pydantic
```

### 4. Configure Gemini API

Create a `.env` file:

``` env
GEMINI_API_KEY=your_api_key_here
```

Do **not** commit `.env` to Git.

Make sure `.gitignore` contains:

``` text
.env
__pycache__/
*.pyc
```

------------------------------------------------------------------------

## ▶️ Running the Application

Start Streamlit:

``` bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

------------------------------------------------------------------------

## 📱 How to Use

### Step 1 --- Export a WhatsApp Chat

Export a WhatsApp conversation as a `.txt` file.

The exported file should contain the standard WhatsApp chat format.

### Step 2 --- Upload the File

Upload the `.txt` file using the sidebar.

### Step 3 --- Select Analysis Scope

Choose:

``` text
Overall
```

or a specific participant.

### Step 4 --- View Chat Analytics

Explore the existing statistics and visualizations.

### Step 5 --- Run AI Analysis

Click:

``` text
🔍 Analyze Conversation with AI
```

The system extracts actionable information.

### Step 6 --- Explore the Action Dashboard

Review:

-   High-priority items
-   Deadlines
-   Tasks
-   Requests
-   Decisions

Use the filters to narrow the results.

### Step 7 --- Verify an AI Result

Open:

``` text
🔎 View source
```

to see the evidence and surrounding conversation.

### Step 8 --- Ask the Conversation

Enter a natural-language question such as:

``` text
When do I need to submit the assignment?
```

and search the conversation.

------------------------------------------------------------------------

## 🔐 Privacy & Security

ChatMind processes the uploaded conversation for analysis.

Important considerations:

-   Keep API keys in `.env`.
-   Never commit API keys to GitHub.
-   Avoid uploading sensitive/private conversations to services unless
    you understand the relevant privacy implications.
-   The application should be used only with conversations that you are
    authorized to analyze.

------------------------------------------------------------------------

## 🎯 Example Use Cases

ChatMind can be useful for conversations where important information
is mixed with casual messages.

### Student Groups

Find:

-   Assignment deadlines
-   Questions to solve
-   Notes/resources people promised to send
-   Decisions about projects
-   Exam-related information

### Project Teams

Find:

-   Assigned tasks
-   Requests between teammates
-   Submission deadlines
-   Project decisions
-   Important technical discussions

### Event / Organization Groups

Find:

-   Responsibilities
-   Pending actions
-   Dates and deadlines
-   Decisions
-   Requests for resources

------------------------------------------------------------------------

## 🧪 Testing

The project includes separate components for testing different parts of
the pipeline.

The AI pipeline has been tested during development using exported WhatsApp conversations. A synthetic example of the expected result is:

``` text
A large WhatsApp conversation
A filtered subset of potentially relevant messages
Structured AI results
```

The system successfully produced structured results containing:

``` text
message_id
date
user
message
candidate_score
candidate_reason
is_important
type
task
deadline
priority
confidence
evidence
```

A representative extracted result was:

``` text
Message ID: synthetic-example

Type:
deadline

Task:
Submit assignment

Deadline:
tomorrow

Priority:
high

Confidence:
0.95

Evidence:
"submit the assignment by tomorrow"
```

------------------------------------------------------------------------

## ⚠️ Current Limitations

ChatMind is an AI-assisted system and extraction quality can depend
on conversational context.

Current limitations include:

-   Ambiguous messages can be difficult to classify as tasks or
    requests.
-   Informal Hinglish and abbreviations can affect extraction.
-   Deadline expressions such as "kal", "1st", or "7 baje" may require
    contextual interpretation.
-   Semantic search quality depends on the embedding model.
-   AI analysis requires access to the configured Gemini API.
-   The current system is primarily designed for exported WhatsApp
    `.txt` conversations rather than real-time WhatsApp integration.

These limitations are areas for future improvement rather than
assumptions that every conversation can be interpreted perfectly.

------------------------------------------------------------------------

## 🔮 Future Improvements

Potential future extensions include:

-   Better Task vs Request classification
-   Improved multilingual/Hinglish understanding
-   More advanced action deduplication
-   User-specific task assignment
-   Calendar export for extracted deadlines
-   Priority customization
-   Better temporal/date normalization
-   Real-time or automated chat ingestion
-   Conversation summarization
-   Improved semantic retrieval
-   Deployment and hosted usage
-   Evaluation dataset and quantitative extraction metrics

------------------------------------------------------------------------

## 💡 What Makes This Different?

Traditional WhatsApp chat analyzers primarily answer:

> **"What happened in this conversation?"**

ChatMind aims to answer:

> **"What useful information can I take from this conversation?"**

The system combines:

``` text
Chat Analytics
      +
Rule-based Candidate Filtering
      +
LLM Information Extraction
      +
Evidence Validation
      +
Action Dashboard
      +
Semantic Search
```

This makes the project more than a visualization tool: it is a prototype
for **conversation intelligence**.

------------------------------------------------------------------------

## 👨‍💻 Author

**Tushar**

B.Tech --- Computer Science & Engineering\
National Institute of Technology Raipur

GitHub: `TusharC19`

------------------------------------------------------------------------

## 📄 License

This project is intended for educational and portfolio purposes.
