"""Streamlit UI for the LLM API lab assistant."""

import os

import streamlit as st
from openai import OpenAI

from template import (
    OPENAI_MINI_MODEL,
    OPENAI_MODEL,
    count_tokens,
    estimate_cost,
    retry_with_backoff,
)

st.set_page_config(
    page_title="LLM Study Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_PERSONA = (
    "Bạn là trợ giảng thân thiện của khóa AI. "
    "Trả lời ngắn gọn, rõ ràng bằng tiếng Việt và dùng ví dụ đơn giản."
)


def init_state() -> None:
    defaults = {
        "messages": [],
        "num_turns": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.num_turns = 0
    st.session_state.total_tokens = 0
    st.session_state.total_cost = 0.0


def get_stream(model: str, persona: str, messages: list[dict]):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context = messages[-6:]
    request_messages = [
        {"role": "system", "content": persona},
        *context,
    ]
    return retry_with_backoff(
        lambda: client.chat.completions.create(
            model=model,
            messages=request_messages,
            stream=True,
        )
    )


init_state()

with st.sidebar:
    st.title("💬 Study Assistant")
    st.caption("LLM API Foundation · Day 1")

    st.subheader("Cấu hình")
    persona = st.text_area(
        "Persona / system prompt",
        value=DEFAULT_PERSONA,
        height=130,
        help="Định hướng vai trò, ngôn ngữ và phong cách trả lời của trợ lý.",
    )
    model = st.selectbox(
        "Model",
        options=[OPENAI_MODEL, OPENAI_MINI_MODEL],
        index=0,
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Thấp hơn = ổn định hơn; cao hơn = đa dạng hơn.",
    )
    max_tokens = st.slider(
        "Max output tokens",
        min_value=64,
        max_value=1024,
        value=256,
        step=64,
    )

    st.divider()
    st.subheader("Thống kê phiên")
    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("Lượt chat", st.session_state.num_turns)
    metric_col_2.metric("Tokens", st.session_state.total_tokens)
    st.metric("Chi phí ước tính", f"${st.session_state.total_cost:.6f}")

    if st.button("🗑️ Xóa cuộc trò chuyện", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()
    st.caption("API key được đọc từ biến môi trường; không hiển thị trong UI.")

st.title("Trợ lý học LLM API")
st.write("Thử nghiệm persona, temperature và hội thoại streaming trong một giao diện trực quan.")

if not st.session_state.messages:
    st.info("Bắt đầu bằng một câu hỏi bên dưới, ví dụ: *Token trong LLM là gì?*")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_parts = []
        try:
            stream = get_stream(model, persona, st.session_state.messages)
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                response_parts.append(content)
                response_placeholder.markdown("".join(response_parts) + "▌")
            reply = "".join(response_parts)
            response_placeholder.markdown(reply)
        except Exception as error:
            st.session_state.messages.pop()
            st.error(
                "Không thể gọi model. Hãy kiểm tra API key, endpoint và tên model."
            )
            st.caption(f"Chi tiết kỹ thuật: {error}")
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": reply}
            )
            costs = estimate_cost(prompt, reply, model)
            st.session_state.num_turns += 1
            st.session_state.total_tokens += count_tokens(prompt, model)
            st.session_state.total_tokens += count_tokens(reply, model)
            st.session_state.total_cost += costs["total_cost"]
