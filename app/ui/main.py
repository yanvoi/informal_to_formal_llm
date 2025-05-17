import requests
import streamlit as st
from streamlit_chat import message


class UI:
    """
    A class to handle the Streamlit user interface for the application.
    """

    def __init__(self):
        pass

    def _get_user_input(self) -> str:
        """
        Get the user's input from the text input field.

        Returns:
            The user's input.
        """

        st.write("Type your text to formalize below:")
        with st.form(key="user_input_form", clear_on_submit=True):
            user_input = st.text_input(
                "Type your text to formalize here:",
                value=st.session_state.get("transcription", ""),
                key="input",
            )
            submitted = st.form_submit_button("Submit")
        if submitted and user_input:
            st.session_state.transcription = ""
            return user_input

    @staticmethod
    def _display_conversation(history: st.session_state):
        """
        Display the conversation history in the UI.

        Args:
            history: The conversation history to display.
        """
        for i, (past, generated) in enumerate(
            zip(history["past"], history["generated"])
        ):
            message(past, is_user=True, key=f"{i}_user")
            message(generated, key=f"{i}")

    def _get_api_response(self, user_input: str) -> dict:
        """
        Get the API response for the user's input.

        Args:
            user_input: The user's input.

        Returns:
            The API response.
        """
        response = requests.post(
            "http://localhost:8000/formalize", json={"text": user_input}
        )
        return response.json()

    def show_main_page(self):
        """
        Display the main page of the application.
        """
        st.title("text formalization 🌐️")

        if "generated" not in st.session_state:
            st.session_state["generated"] = []
        if "past" not in st.session_state:
            st.session_state["past"] = []
        if "transcription" not in st.session_state:
            st.session_state["transcription"] = ""

        self._display_conversation(st.session_state)

        user_input = self._get_user_input()

        if user_input:
            response = self._get_api_response(user_input)
            output = {"answer": response["formalized_text"]}
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output["answer"])

            st.rerun()

    def main(self):
        self.show_main_page()


if __name__ == "__main__":
    ui = UI()
    ui.main()
