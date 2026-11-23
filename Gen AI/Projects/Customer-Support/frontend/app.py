import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/api/v1/support/tool-chat/stream"


st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖",
    layout="centered",
)


st.title("🤖 AI Customer Support")
st.caption("FastAPI + Ollama + Qwen3 + Tool Calling")


if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_message = st.chat_input(
    "How can I help you?"
)


if user_message:

    # Display user message immediately
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_message)


    # Assistant response
    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""


        try:

            with requests.post(
                API_URL,
                json={
                    "message": user_message
                },
                stream=True,
                timeout=90,
            ) as response:

                response.raise_for_status()


                for chunk in response.iter_content(
                    chunk_size=None,
                    decode_unicode=True,
                ):

                    if not chunk:
                        continue

                    full_response += chunk

                    response_placeholder.markdown(
                        full_response
                    )


        except requests.exceptions.Timeout:

            full_response = (
                "The AI service took too long to respond."
            )

            response_placeholder.error(
                full_response
            )


        except requests.exceptions.ConnectionError:

            full_response = (
                "Unable to connect to the support service. "
                "Please make sure the FastAPI server is running."
            )

            response_placeholder.error(
                full_response
            )


        except requests.exceptions.RequestException as exc:

            full_response = (
                "The support service encountered an error."
            )

            response_placeholder.error(
                full_response
            )


        except Exception:

            full_response = (
                "An unexpected error occurred."
            )

            response_placeholder.error(
                full_response
            )


    if full_response:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )