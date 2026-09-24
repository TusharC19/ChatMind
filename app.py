import re
import streamlit as st
import pandas as pd
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns

from ai.analyzer import analyze_chat
from ai.semantic_search import SemanticSearch


def show_action_source(df, row):
    message_id = int(row["message_id"])

    matching = df.index[
        df["message_id"] == message_id
    ].tolist()

    if not matching:
        st.warning("Original message not found.")
        return

    current_index = matching[0]

    start = max(0, current_index - 2)
    end = min(len(df), current_index + 3)

    context_df = df.iloc[start:end]

    st.markdown("**Evidence:**")
    evidence = row.get("evidence")

    if evidence and not pd.isna(evidence):
        st.info(f'"{evidence}"')
    else:
        st.caption("No evidence available.")

    st.markdown("**Original conversation:**")

    for _, context_row in context_df.iterrows():
        prefix = "👉 " if int(context_row["message_id"]) == message_id else ""

        st.markdown(
            f"{prefix}**{context_row['user']}** "
            f"({context_row['date']})  \n"
            f"{context_row['message']}"
        )

    st.caption(f"Message ID: {message_id}")


def deduplicate_actions(actions_df):
    """
    Merge duplicate actions even when Gemini assigns different types
    (for example, task vs request).

    The action text and deadline are used as the identity of an action.
    The highest-confidence occurrence is retained, so its source/evidence
    remains available.
    """
    if actions_df is None or actions_df.empty:
        return actions_df

    result = actions_df.copy()

    def normalize_action(value):
        if value is None or pd.isna(value):
            return ""

        text = str(value).lower().strip()

        # Remove common punctuation and normalize whitespace.
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\\s+", " ", text).strip()

        # Small normalization for common wording differences.
        replacements = {
            "provided": "",
            "the": "",
            "a": "",
            "an": "",
        }

        words = [
            word for word in text.split()
            if word not in replacements
        ]

        return " ".join(words)

    def normalize_deadline(value):
        if value is None or pd.isna(value):
            return ""
        return " ".join(str(value).lower().strip().split())

    # IMPORTANT:
    # Type is deliberately NOT part of the duplicate key.
    #
    # This means:
    #   task  -> "fill out the assignment form"
    #   request -> "fill out the assignment form"
    #
    # are recognized as the same practical action.
    result["_action_key"] = result.apply(
        lambda row: (
            normalize_action(
                row.get("task")
            ) or normalize_action(
                row.get("message")
            ),
            normalize_deadline(
                row.get("deadline")
            ),
        ),
        axis=1,
    )

    result["_confidence_num"] = pd.to_numeric(
        result["confidence"],
        errors="coerce"
    ).fillna(0)

    # Prefer higher-confidence extraction.
    # If confidence is equal, prefer deadline > task > request > decision.
    type_priority = {
        "deadline": 4,
        "task": 3,
        "request": 2,
        "decision": 1,
    }

    result["_type_priority"] = (
        result["type"]
        .astype(str)
        .str.lower()
        .map(type_priority)
        .fillna(0)
    )

    result = (
        result
        .sort_values(
            ["_action_key", "_confidence_num", "_type_priority"],
            ascending=[True, False, False],
        )
        .drop_duplicates(
            subset="_action_key",
            keep="first",
        )
        .drop(
            columns=[
                "_action_key",
                "_confidence_num",
                "_type_priority",
            ]
        )
    )

    if "message_id" in result.columns:
        result = result.sort_values("message_id")

    return result.reset_index(drop=True)



def generate_grounded_answer(search_results):
    """
    Generate a simple answer from the highest-ranked
    AI-supported search result.

    No additional LLM/API call is used.
    """

    if not search_results:
        return None

    for result in search_results:

        ai_result = result.get("ai_result")

        if ai_result is None:
            continue

        result_type = ai_result.get("type")
        task = ai_result.get("task")
        deadline = ai_result.get("deadline")
        evidence = ai_result.get("evidence")

        if result_type == "deadline":

            if task and deadline:
                answer = (
                    f"You need to **{task}** "
                    f"**{deadline}**."
                )

            elif deadline:
                answer = (
                    f"The relevant deadline is "
                    f"**{deadline}**."
                )

            else:
                answer = (
                    "A deadline was found in the conversation."
                )

            return {
                "answer": answer,
                "result": result,
                "evidence": evidence
            }

        if result_type == "task":

            if task:
                answer = (
                    f"The conversation indicates that "
                    f"you need to **{task}**."
                )
            else:
                answer = (
                    "A task was identified in the conversation."
                )

            return {
                "answer": answer,
                "result": result,
                "evidence": evidence
            }

        if result_type == "request":

            if task:
                answer = (
                    f"The conversation contains a request "
                    f"related to **{task}**."
                )
            else:
                answer = (
                    "A request related to your question "
                    "was found."
                )

            return {
                "answer": answer,
                "result": result,
                "evidence": evidence
            }

        if result_type == "decision":

            answer = (
                "A decision related to your question "
                "was found in the conversation."
            )

            return {
                "answer": answer,
                "result": result,
                "evidence": evidence
            }

    return None


# ==================================================
# SESSION STATE
# ==================================================

if "ai_results" not in st.session_state:
    st.session_state.ai_results = None

if "show_analysis" not in st.session_state:
    st.session_state.show_analysis = False

if "semantic_search" not in st.session_state:
    st.session_state.semantic_search = None

if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = None


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader(
    "Choose a file"
)


# ==================================================
# PROCESS UPLOADED CHAT
# ==================================================

if uploaded_file is not None:

    # --------------------------------------------------
    # Detect new uploaded file
    # --------------------------------------------------

    current_file_key = (
        uploaded_file.name,
        uploaded_file.size
    )

    if (
        st.session_state.uploaded_file_key
        != current_file_key
    ):

        st.session_state.uploaded_file_key = (
            current_file_key
        )

        st.session_state.ai_results = None
        st.session_state.semantic_search = None
        st.session_state.show_analysis = False


    # --------------------------------------------------
    # Read file
    # --------------------------------------------------

    bytes_data = uploaded_file.getvalue()

    data = bytes_data.decode("utf-8")


    # --------------------------------------------------
    # Preprocess chat
    # --------------------------------------------------

    df = preprocessor.preprocess(data)


    # ==================================================
    # USER SELECTION
    # ==================================================

    user_list = df["user"].unique().tolist()

    if "group_notification" in user_list:
        user_list.remove("group_notification")

    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox(
        "Show analysis wrt",
        user_list
    )


    # ==================================================
    # SHOW ANALYSIS
    # ==================================================

    if st.sidebar.button("Show Analyze"):
        st.session_state.show_analysis = True


    # ==================================================
    # EXISTING CHAT ANALYTICS
    # ==================================================

    if st.session_state.show_analysis:

        # ==============================================
        # TOP STATISTICS
        # ==============================================

        num_messages, words, num_media_messages, num_links = (
            helper.fetch_stats(
                selected_user,
                df
            )
        )

        st.title("Top Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total Words")
            st.title(words)

        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)

        with col4:
            st.header("Links Shared")
            st.title(num_links)


        # ==============================================
        # MONTHLY TIMELINE
        # ==============================================

        st.title("Monthly TimeLine")

        timeline = helper.monthly_timeline(
            selected_user,
            df
        )

        fig, ax = plt.subplots()

        ax.plot(
            timeline["time"],
            timeline["message"]
        )

        plt.xticks(rotation="vertical")

        st.pyplot(fig)


        # ==============================================
        # DAILY TIMELINE
        # ==============================================

        st.title("Daily Timeline")

        daily_timeline = helper.daily_timeline(
            selected_user,
            df
        )

        fig, ax = plt.subplots()

        ax.plot(
            daily_timeline["only_date"],
            daily_timeline["message"],
            color="black"
        )

        plt.xticks(rotation="vertical")

        st.pyplot(fig)


        # ==============================================
        # ACTIVITY MAP
        # ==============================================

        st.title("Activity Map")

        col1, col2 = st.columns(2)

        with col1:

            st.header("Most busy day")

            busy_day = helper.week_activity_map(
                selected_user,
                df
            )

            fig, ax = plt.subplots()

            ax.bar(
                busy_day.index,
                busy_day.values,
                color="purple"
            )

            plt.xticks(rotation="vertical")

            st.pyplot(fig)


        with col2:

            st.header("Most busy month")

            busy_month = helper.month_activity_map(
                selected_user,
                df
            )

            fig, ax = plt.subplots()

            ax.bar(
                busy_month.index,
                busy_month.values,
                color="orange"
            )

            plt.xticks(rotation="vertical")

            st.pyplot(fig)


        # ==============================================
        # WEEKLY ACTIVITY MAP
        # ==============================================

        st.title("Weekly Activity Map")

        user_heatmap = helper.activity_heatmap(
            selected_user,
            df
        )

        fig, ax = plt.subplots()

        sns.heatmap(user_heatmap)

        st.pyplot(fig)


        # ==============================================
        # MOST BUSY USERS
        # ==============================================

        if selected_user == "Overall":

            st.title("Most Busy Users")

            x, new_df = helper.fetch_most_busy_users(df)

            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:

                ax.bar(
                    x.index,
                    x.values,
                    color="red"
                )

                plt.xticks(rotation="vertical")

                st.pyplot(fig)

            with col2:

                st.dataframe(new_df)


        # ==============================================
        # WORD CLOUD
        # ==============================================

        st.title("Wordcloud")

        df_wc = helper.create_wordcloud(
            selected_user,
            df
        )

        fig, ax = plt.subplots()

        ax.imshow(df_wc)

        ax.axis("off")

        st.pyplot(fig)


        # ==============================================
        # MOST COMMON WORDS
        # ==============================================

        most_common_df = helper.most_common_words(
            selected_user,
            df
        )

        fig, ax = plt.subplots()

        ax.barh(
            most_common_df[0],
            most_common_df[1],
            color="green"
        )

        plt.xticks(rotation="vertical")

        st.title("Most Common Words")

        st.pyplot(fig)


        # ==============================================
        # EMOJI ANALYSIS
        # ==============================================

        emoji_df = helper.emoji_helper(
            selected_user,
            df
        )

        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.dataframe(emoji_df)

        with col2:

            if (
                emoji_df is not None
                and not emoji_df.empty
            ):

                fig, ax = plt.subplots()

                top_emojis = emoji_df.head()

                ax.pie(
                    top_emojis[1],
                    labels=top_emojis[0],
                    autopct="%1.1f%%",
                    startangle=90
                )

                ax.axis("equal")

                st.pyplot(fig)

            else:

                st.write(
                    "No emojis found in chat."
                )


        # ==================================================
        # AI CONVERSATION INTELLIGENCE
        # ==================================================

        st.title("🧠 Conversation Intelligence")

        st.write(
            "Extract actionable information from "
            "your WhatsApp conversations using AI."
        )


        # ==============================================
        # AI ANALYSIS BUTTON
        # ==============================================

        if st.button(
            "🔍 Analyze Conversation with AI"
        ):

            with st.spinner(
                "Analyzing conversation... "
                "This may take a moment."
            ):

                try:

                    # ----------------------------------
                    # Run AI analysis
                    # ----------------------------------

                    ai_results = analyze_chat(
                        df,
                        window=2,
                        batch_size=5,
                        delay=1
                    )

                    st.session_state.ai_results = (
                        ai_results
                    )


                    # ----------------------------------
                    # Build semantic search index
                    # ----------------------------------

                    with st.spinner(
                        "Building conversation search index..."
                    ):

                        semantic_search = SemanticSearch()

                        semantic_search.build_index(
                            df,
                            ai_results=ai_results
                        )

                        st.session_state.semantic_search = (
                            semantic_search
                        )


                    st.success(
                        f"Analysis complete! "
                        f"{len(ai_results)} messages analyzed."
                    )

                except Exception as error:

                    st.error(
                        f"AI analysis failed: {error}"
                    )


        # ==============================================
        # DISPLAY AI RESULTS
        # ==============================================

        if st.session_state.ai_results is not None:

            ai_results = (
                st.session_state.ai_results
            )

            actionable_results = deduplicate_actions(
                ai_results
            )


            # ==========================================
            # AI SUMMARY
            # ==========================================

            st.subheader("📋 Extracted Insights")

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "AI Results",
                    len(ai_results)
                )

            with col2:

                task_count = len(
                    ai_results[
                        ai_results["type"] == "task"
                    ]
                )

                st.metric(
                    "Tasks",
                    task_count
                )

            with col3:

                request_count = len(
                    ai_results[
                        ai_results["type"] == "request"
                    ]
                )

                st.metric(
                    "Requests",
                    request_count
                )

            with col4:

                deadline_count = len(
                    ai_results[
                        ai_results["type"] == "deadline"
                    ]
                )

                st.metric(
                    "Deadlines",
                    deadline_count
                )


            # ==========================================
            # ACTIONABLE MESSAGES
            # ==========================================

            st.subheader(
                "Actionable Messages"
            )

            actionable = actionable_results[
                actionable_results["type"].isin(
                    [
                        "task",
                        "request",
                        "deadline",
                        "decision"
                    ]
                )
            ].copy()


            if not actionable.empty:

                display_columns = [
                    "message_id",
                    "user",
                    "message",
                    "type",
                    "task",
                    "deadline",
                    "priority",
                    "confidence",
                    "evidence"
                ]

                st.dataframe(
                    actionable[display_columns],
                    width="stretch"
                )

            else:

                st.info(
                    "No actionable information was found."
                )

            # ==================================================
            # ACTION DASHBOARD
            # ==================================================

            st.subheader("🎯 Action Dashboard")

            if len(actionable_results) < len(ai_results):
                st.caption(
                    f"Showing {len(actionable_results)} unique actions "
                    f"from {len(ai_results)} AI results. "
                    "Similar actions across different AI types are merged."
                )

            st.write(
                "Important tasks, requests, deadlines, "
                "and decisions identified from your conversation."
            )


            # ==============================================
            # FILTERS
            # ==============================================

            filter_col1, filter_col2 = st.columns(2)

            with filter_col1:

                action_type = st.selectbox(
                    "Filter by type",
                    [
                        "All",
                        "Tasks",
                        "Requests",
                        "Deadlines",
                        "Decisions"
                    ],
                    key="action_type_filter"
                )

            with filter_col2:

                priority_filter = st.selectbox(
                    "Filter by priority",
                    [
                        "All",
                        "High",
                        "Medium",
                        "Low"
                    ],
                    key="priority_filter"
                )


            # ==============================================
            # APPLY FILTERS
            # ==============================================

            dashboard_df = actionable_results.copy()

            type_mapping = {
                "Tasks": "task",
                "Requests": "request",
                "Deadlines": "deadline",
                "Decisions": "decision"
            }

            if action_type != "All":

                dashboard_df = dashboard_df[
                    dashboard_df["type"]
                    == type_mapping[action_type]
                ]

            if priority_filter != "All":

                dashboard_df = dashboard_df[
                    dashboard_df["priority"].fillna("").str.lower()
                    == priority_filter.lower()
                ]


            # ==============================================
            # HIGH PRIORITY
            # ==============================================

            high_priority = dashboard_df[
                dashboard_df["priority"].fillna("").str.lower()
                == "high"
            ].copy()

            if not high_priority.empty:

                st.markdown("### 🔴 High Priority")

                for _, row in high_priority.iterrows():

                    task_text = row["task"]

                    if (
                        task_text is None
                        or pd.isna(task_text)
                    ):
                        task_text = row["message"]

                    deadline = row["deadline"]

                    if (
                        deadline is None
                        or pd.isna(deadline)
                    ):
                        deadline = "Not specified"

                    st.warning(
                        f"**{task_text}**\n\n"
                        f"Type: {row['type']}  |  "
                        f"Deadline: {deadline}  |  "
                        f"Confidence: {row['confidence']}"
                    )

                    with st.expander(
                        f"🔎 View source — Message {int(row['message_id'])}"
                    ):
                        show_action_source(df, row)

            # High-priority actions have already been displayed above.
            # Exclude them from the category sections so one action
            # appears only once in the dashboard.
            high_priority_ids = set(
                high_priority["message_id"].tolist()
            )

            remaining_df = dashboard_df[
                ~dashboard_df["message_id"].isin(high_priority_ids)
            ].copy()


            # ==============================================
            # DEADLINES
            # ==============================================

            deadline_df = remaining_df[
                remaining_df["type"] == "deadline"
            ]

            if not deadline_df.empty:

                st.markdown("### 📅 Deadlines")

                for _, row in deadline_df.iterrows():

                    task_text = row["task"]

                    if (
                        task_text is None
                        or pd.isna(task_text)
                    ):
                        task_text = "Deadline mentioned"

                    deadline = row["deadline"]

                    if (
                        deadline is None
                        or pd.isna(deadline)
                    ):
                        deadline = "Date not specified"

                    st.write(
                        f"📌 **{task_text}** → "
                        f"**{deadline}**"
                    )

                    with st.expander(
                        f"🔎 View source — Message {int(row['message_id'])}"
                    ):
                        show_action_source(df, row)


            # ==============================================
            # TASKS
            # ==============================================

            task_df = remaining_df[
                remaining_df["type"] == "task"
            ]

            if not task_df.empty:

                st.markdown("### ✅ Tasks")

                for _, row in task_df.iterrows():

                    task_text = row["task"]

                    if (
                        task_text is None
                        or pd.isna(task_text)
                    ):
                        task_text = row["message"]

                    st.write(
                        f"• **{task_text}**"
                    )

                    with st.expander(
                        f"🔎 View source — Message {int(row['message_id'])}"
                    ):
                        show_action_source(df, row)


            # ==============================================
            # REQUESTS
            # ==============================================

            request_df = remaining_df[
                remaining_df["type"] == "request"
            ]

            if not request_df.empty:

                st.markdown("### 🙋 Requests")

                for _, row in request_df.iterrows():

                    task_text = row["task"]

                    if (
                        task_text is None
                        or pd.isna(task_text)
                    ):
                        task_text = row["message"]

                    st.write(
                        f"• **{task_text}**"
                    )


            # ==============================================
            # DECISIONS
            # ==============================================

            decision_df = remaining_df[
                remaining_df["type"] == "decision"
            ]

            if not decision_df.empty:

                st.markdown("### 🤝 Decisions")

                for _, row in decision_df.iterrows():

                    task_text = row["task"]

                    if (
                        task_text is None
                        or pd.isna(task_text)
                    ):
                        task_text = row["message"]

                    st.write(
                        f"• **{task_text}**"
                    )


            # ==============================================
            # NO RESULTS
            # ==============================================

            if dashboard_df.empty:

                st.info(
                    "No items match the selected filters."
                )

        # ==================================================
        # ASK YOUR CONVERSATION
        # ==================================================

        st.title("🔎 Ask Your Conversation")

        st.write(
            "Search your conversation using natural language."
        )


        query = st.text_input(
            "What do you want to find?",
            placeholder=(
                "e.g. When do I need to "
                "submit the assignment?"
            )
        )


        if st.button(
            "Search Conversation"
        ):

            if not query.strip():

                st.warning(
                    "Please enter a question or search query."
                )

            elif (
                st.session_state.semantic_search
                is None
            ):

                st.warning(
                    "Please analyze the conversation first."
                )

            else:

                with st.spinner(
                    "Searching conversation..."
                ):

                    search_results = (
                        st.session_state.semantic_search.search(
                            query,
                            top_k=5
                        )
                    )


                if not search_results:

                    st.info(
                        "No relevant messages found."
                    )

                else:

                    # ==============================================
                    # GROUNDED ANSWER
                    # ==============================================

                    grounded_answer = generate_grounded_answer(
                        search_results
                    )

                    if grounded_answer is not None:

                        st.subheader("🤖 Answer")

                        st.markdown(
                            grounded_answer["answer"]
                        )

                        result = grounded_answer["result"]

                        st.caption(
                            f"Based on Message {result['message_id']}"
                        )

                        evidence = grounded_answer["evidence"]

                        if evidence:
                            st.info(
                                f'📌 Evidence: "{evidence}"'
                            )

                        ai_result = result.get("ai_result")

                        if ai_result:

                            answer_col1, answer_col2 = st.columns(2)

                            with answer_col1:
                                st.write(
                                    f"**Priority:** "
                                    f"{ai_result.get('priority')}"
                                )

                            with answer_col2:
                                st.write(
                                    f"**Confidence:** "
                                    f"{ai_result.get('confidence')}"
                                )

                        st.divider()

                    # ==============================================
                    # SUPPORTING MESSAGES
                    # ==============================================

                    st.subheader("🔍 Relevant Messages")

                    for result in search_results:

                        st.markdown(
                            f"### Message "
                            f"{result['message_id']}"
                        )

                        st.write(
                            f"**{result['user']}:** "
                            f"{result['message']}"
                        )


                        # ----------------------------------
                        # SEARCH SCORES
                        # ----------------------------------

                        score_col1, score_col2, score_col3 = (
                            st.columns(3)
                        )

                        with score_col1:

                            st.caption(
                                f"Semantic: "
                                f"{result['semantic_score']:.4f}"
                            )

                        with score_col2:

                            st.caption(
                                f"Keyword: "
                                f"{result['keyword_score']:.4f}"
                            )

                        with score_col3:

                            st.caption(
                                f"AI relevance: "
                                f"{result['ai_score']:.4f}"
                            )


                        # ----------------------------------
                        # AI INSIGHT
                        # ----------------------------------

                        ai_result = result.get(
                            "ai_result"
                        )


                        if ai_result is not None:

                            st.markdown(
                                "#### 🧠 AI Insight"
                            )

                            insight_col1, insight_col2 = (
                                st.columns(2)
                            )

                            with insight_col1:

                                st.write(
                                    f"**Type:** "
                                    f"{ai_result['type']}"
                                )

                                task = ai_result.get(
                                    "task"
                                )

                                if task is not None:

                                    st.write(
                                        f"**Task:** {task}"
                                    )

                                deadline = ai_result.get(
                                    "deadline"
                                )

                                if deadline is not None and not pd.isna(deadline):

                                    st.write(
                                        f"**Deadline:** "
                                        f"{deadline}"
                                    )

                            with insight_col2:

                                st.write(
                                    f"**Priority:** "
                                    f"{ai_result['priority']}"
                                )

                                st.write(
                                    f"**Confidence:** "
                                    f"{ai_result['confidence']}"
                                )

                                evidence = ai_result.get(
                                    "evidence"
                                )

                                if evidence is not None:

                                    st.write(
                                        f"**Evidence:** "
                                        f"{evidence}"
                                    )


                        # ----------------------------------
                        # ORIGINAL CONTEXT
                        # ----------------------------------

                        with st.expander(
                            "💬 View conversation context"
                        ):

                            for context_message in (
                                result["context"]
                            ):

                                st.write(
                                    f"**[{context_message['message_id']}] "
                                    f"{context_message['user']}:** "
                                    f"{context_message['message']}"
                                )


                        st.divider()