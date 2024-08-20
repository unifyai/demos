import streamlit as st
from unify import AsyncUnify
from unify.exceptions import UnifyError
import asyncio
import pandas as pd
import json
import requests
import random
import helpers

st.set_page_config(
    page_title="Chatbot Arena",
    page_icon="🤖",
    layout="wide",
)


def select_model(api_key: str = "", authenticated: bool = False) -> None:
    """Select two models for the Unify API. The models are picked through
    selectbox options.

    Parameters
    ----------
    api_key
        The key to the Unify API. If none is provided, the UI of the app remains disabled.
        Default: "".
    authenticated
        The return value of the request call to the Unify API. If API key authentication returns
        an error or is not assigned to any user, this parameters disables the app's UI.
        Default: False

    Returns
    -------
    None
    """

    global json_data, all_models, data

    disabled = not (bool(api_key) and bool(authenticated))
    model1_other_disabled = True
    model2_other_disabled = True

    st.session_state.model1_selectbox = st.selectbox(
        "Select the first model's endpoint:",
        options=all_models + ["I'm feeling lucky"],
        disabled=disabled,
        index=st.session_state.index_model1,
        on_change=lambda: (
            setattr(st.session_state, "chat_history1", []),
            setattr(st.session_state, "chat_history2", []),
            setattr(st.session_state, "winner_selected", True),
            setattr(st.session_state, "prompt_provided", False),
            setattr(st.session_state, "new_models_selected", True),
        ),
    )
    if st.session_state.model1_selectbox == "I'm feeling lucky":
        st.session_state.model1_selectbox = random.choice(all_models[:-1])
    st.session_state.index_model1 = all_models.index(st.session_state.model1_selectbox)

    if st.session_state.model1_selectbox == "other":
        model1_other_disabled = False
    st.text_input(
        'If "other", provide your own model:',
        placeholder="<model>@<provider>",
        disabled=model1_other_disabled,
        value=st.session_state.value_model1_other,
        on_change=lambda: (
            setattr(st.session_state, "chat_history1", []),
            setattr(st.session_state, "chat_history2", []),
            setattr(st.session_state, "winner_selected", True),
            setattr(st.session_state, "prompt_provided", False),
            setattr(st.session_state, "new_models_selected", True),
        ),
        key="model1_other",
    )
    st.session_state.model2_selectbox = st.selectbox(
        "Select the second model's endpoint:",
        options=all_models + ["I'm feeling lucky"],
        disabled=disabled,
        index=st.session_state.index_model2,
        on_change=lambda: (
            setattr(st.session_state, "chat_history1", []),
            setattr(st.session_state, "chat_history2", []),
            setattr(st.session_state, "winner_selected", True),
            setattr(st.session_state, "prompt_provided", False),
            setattr(st.session_state, "new_models_selected", True),
        ),
    )
    if st.session_state.model2_selectbox == "I'm feeling lucky":
        st.session_state.model2_selectbox = random.choice(all_models[:-1])
    st.session_state.index_model2 = all_models.index(st.session_state.model2_selectbox)

    if st.session_state.model2_selectbox == "other":
        model2_other_disabled = False
    st.text_input(
        'If "other", provide your own model:',
        placeholder="<model>@<provider>",
        disabled=model2_other_disabled,
        value=st.session_state.value_model2_other,
        on_change=lambda: (
            setattr(st.session_state, "chat_history1", []),
            setattr(st.session_state, "chat_history2", []),
            setattr(st.session_state, "winner_selected", True),
            setattr(st.session_state, "prompt_provided", False),
            setattr(st.session_state, "new_models_selected", True),
        ),
        key="model2_other",
    )
    selected_model1 = (
        st.session_state.model1_selectbox
        if st.session_state.model1_selectbox != "other"
        else st.session_state.model1_other
    )
    selected_model2 = (
        st.session_state.model2_selectbox
        if st.session_state.model2_selectbox != "other"
        else st.session_state.model2_other
    )

    # st.session_state.index_model1 = all_models.index(st.session_state.model1_selectbox)
    # st.session_state.index_model2 = all_models.index(st.session_state.model2_selectbox)
    if st.session_state.model1_selectbox == "other":
        st.session_state.value_model1_other = selected_model1
    if st.session_state.model2_selectbox == "other":
        st.session_state.value_model2_other = selected_model2

    selected_models = [selected_model1, selected_model2]
    random.shuffle(selected_models)

    if st.session_state.new_models_selected in [True, None]:
        st.session_state["model1"] = selected_models.pop(0)
        st.session_state["model2"] = selected_models.pop(0)
        st.session_state.new_models_selected = False
        st.rerun()


def history(model: str = "model1", output: str = "") -> None:
    """Assign to session states and manage the contents of the chat history.

    Parameters
    ----------
    model
        the key of the model. Takes on the values in ["model1", "model2"].
        Default: "model1".
    output
        the response of the model to the prompt.
        Default: ""

    Returns
    -------
    None
    """

    if model == "model1":
        st.session_state["chat_history1"].append(
            {"role": "assistant", "content": output}
        )
    elif model == "model2":
        st.session_state["chat_history2"].append(
            {"role": "assistant", "content": output}
        )

    else:
        st.write("Please, enter the model1 or model2 in history function.")
    if len(st.session_state["chat_history1"]) >= 10:
        st.session_state["chat_history1"].pop(0)
    if len(st.session_state["chat_history2"]) >= 10:
        st.session_state["chat_history2"].pop(0)


def input_api_key(api_key: str = " ") -> None:
    """Authorize the Unify API key provided by the user. If the key is
    recognised, the app's UI will be unlocked.

    Parameters
    ----------
    api_key
        The key to the Unify API.
        Default: " ".

    Returns
    -------
    None
    """

    authorisation_url = "https://api.unify.ai/v0/get_credits"
    r = requests.get(
        authorisation_url,
        headers={"accept": "application/json", "Authorization": f"Bearer {api_key}"},
    ).json()

    if "id" in r.keys():
        setattr(st.session_state, "api_key_provided", True)
        setattr(st.session_state, "authenticated", True)
        st.sidebar.write(f"Session user credits: {r['credits']}")
        st.session_state["api_key"] = api_key
    elif "detail" in r.keys():
        setattr(st.session_state, "api_key_provided", False)
        setattr(st.session_state, "authenticated", False)
        st.sidebar.write(f"{r['detail']}")
    elif "error" in r.keys():
        setattr(st.session_state, "api_key_provided", False)
        setattr(st.session_state, "authenticated", False)
        st.sidebar.write(f"{r['error']}")


def call_model(Endpoint: str) -> AsyncUnify:
    """Prepare the Unify model to which the prompts will be sent.

    Parameters
    ----------
    Endpoint
        the endpoint of the LLM model in the format "<model>@<provider>".

    Returns
    -------
    async_unify
        the AsyncUnify object with the endpoint assigned to it.
    """

    async_unify = AsyncUnify(api_key=st.session_state["api_key"], endpoint=Endpoint)
    return async_unify


async def main() -> None:
    """Main loop of the streamlit app. Contains all of the flow control of the
    app and generates UI elements.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    global all_models, data, json_data

    st.set_option("deprecation.showPyplotGlobalUse", False)
    helpers.init_session("keys")
    if st.session_state.source is True:
        source = "online"
    else:
        source = "offline"
    helpers.init_session(source)

    _, theme_col = st.columns([7, 1])
    with theme_col:
        helpers.Buttons.change_theme_button()

    all_models = list(st.session_state.models)
    json_data = st.session_state.leaderboard

    if st.session_state.source is True:
        data = {model: 0 for model in json_data.index}
    else:
        data = pd.read_csv(
            "leaderboard.csv"
        )  # This will raise an error if the file does not exist

    st.session_state.code_input = ""
    st.markdown(
        """
        <h1 style='text-align: center; color: green;'>
            ⚔️ Unify Chatbot Arena: Benchmarking LLMs in the Wild 🚀
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.subheader("Unify API Key")
    api_key = st.sidebar.text_input(
        " ",
        value=st.session_state.api_key,
        placeholder="API key is required to proceed.",
        type="password",
    )
    input_api_key(api_key)

    # Display sidebar widgets
    with st.sidebar:
        select_model(st.session_state.api_key, st.session_state.authenticated)
    col11, col21 = st.columns(2)
    # Display chat UI
    with col11:
        if all([st.session_state.winner_selected, st.session_state.prompt_provided]):
            st.markdown(
                "<span style='font-size:20px; color:blue;'>Model 1: "
                + st.session_state["model1"]
                + "</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='font-size:20px; color:blue;'>Model 1: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░</span>",
                unsafe_allow_html=True,
            )
    with col21:
        if all([st.session_state.winner_selected, st.session_state.prompt_provided]):
            st.markdown(
                "<span style='font-size:20px; color:blue;'>Model 2: "
                + st.session_state["model2"]
                + "</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='font-size:20px; color:blue;'>Model 2: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░</span>",
                unsafe_allow_html=True,
            )
    col1, col2 = st.columns(2)
    with col1:
        cont1 = st.container(height=500)
    with col2:
        cont2 = st.container(height=500)
    if "chat_history1" not in st.session_state:
        st.session_state["chat_history1"] = []
    if "chat_history2" not in st.session_state:
        st.session_state["chat_history2"] = []
    if prompt := st.chat_input(
        "Say something",
        disabled=False if st.session_state.api_key_provided is True else True,
        on_submit=lambda: (
            setattr(st.session_state, "winner_selected", False),
            setattr(st.session_state, "prompt_provided", True),
        ),
    ):
        st.session_state["chat_input"] = prompt
        st.session_state.code_input = prompt
        st.session_state["chat_history1"].append(
            {"role": "user", "content": st.session_state["chat_input"]}
        )
        st.session_state["chat_history2"].append(
            {"role": "user", "content": st.session_state["chat_input"]}
        )
        message1 = st.session_state["chat_history1"]
        message2 = st.session_state["chat_history2"]
        helpers.print_history(contain=(cont1, cont2))
        u1 = None
        u2 = None
        try:
            u1 = call_model(st.session_state["model1"])
            if st.session_state["model1"] not in all_models:
                with open("models.json", "w") as models_file_update:
                    upd_models = [model for model in all_models]
                    upd_models[-1] = st.session_state["model1"]
                    upd_models.append("other")
                    st.session_state.models = upd_models
                    upd_models_dict = {"models": tuple(upd_models)}
                    json.dump(upd_models_dict, models_file_update)
            if (
                model1_to_add := st.session_state["model1"][
                    : st.session_state["model1"].find("@")
                ]
            ) not in data["Model Name"]:
                st.session_state["vote_counts"].at[f"{model1_to_add}", "Wins ⭐"] = 0
                st.session_state["vote_counts"].at[f"{model1_to_add}", "Losses ❌"] = 0
                st.session_state["vote_counts"].at[
                    f"{model1_to_add}", "Model Name"
                ] = f"{model1_to_add}"
        except UnifyError:
            setattr(st.session_state, "winner_selected", True)
            if "@" not in st.session_state["model1"]:
                cont1.error(
                    "Model endpoints have to follow the '<model>@<provider>' format"
                )
                cont2.error(
                    "Model endpoints have to follow the '<model>@<provider>' format"
                )
            else:
                cont1.error("One of the models is not currently supported.")
                cont2.error("One of the models is not currently supported.")
        try:
            u2 = call_model(st.session_state["model2"])
            if st.session_state["model2"] not in all_models:
                with open("models.json", "w") as models_file_update:
                    upd_models = [model for model in all_models]
                    upd_models[-1] = st.session_state["model2"]
                    upd_models.append("other")
                    st.session_state.models = upd_models
                    upd_models_dict = {"models": tuple(upd_models)}
                    json.dump(upd_models_dict, models_file_update)
            if (
                model2_to_add := st.session_state["model2"][
                    : st.session_state["model2"].find("@")
                ]
            ) not in data["Model Name"]:
                st.session_state["vote_counts"].at[f"{model2_to_add}", "Wins ⭐"] = 0
                st.session_state["vote_counts"].at[f"{model2_to_add}", "Losses ❌"] = 0

                st.session_state["vote_counts"].at[
                    f"{model2_to_add}", "Model Name"
                ] = f"{model2_to_add}"
        except UnifyError:
            setattr(st.session_state, "winner_selected", True)
            if "@" not in st.session_state["model2"]:
                cont1.error(
                    "Model endpoints have to follow the '<model>@<provider>' format"
                )
                cont2.error(
                    "Model endpoints have to follow the '<model>@<provider>' format"
                )
            else:
                cont1.error("One of the models is not currently supported.")
                cont2.error("One of the models is not currently supported.")

        async def call(unify_obj, model, contain, message):
            try:
                async_stream = await unify_obj.generate(messages=message, stream=True)
                placeholder = contain.empty()
                full_response = ""
                async for chunk in async_stream:
                    full_response += chunk
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
                if full_response == "":
                    full_response = "<No response>"
            except UnifyError as error_message:
                contain.error(
                    f"The selected model and/or provider might not be available. Clearing the chat history.\n {error_message}",
                    icon="🚨",
                )
                if model == "model1":
                    st.session_state.chat_history1 = []
                if model == "model2":
                    st.session_state.chat_history2 = []
                setattr(st.session_state, "winner_selected", False)
            except IndexError as error_message:
                contain.error(
                    f"There was an issue with the model's response:\n {error_message}",
                    icon="🚨",
                )
                setattr(st.session_state, "winner_selected", False)
            finally:
                history(model=model, output=full_response)

        await asyncio.gather(
            call(u1, model="model1", contain=cont1, message=message1),
            call(u2, model="model2", contain=cont2, message=message2),
        )

    c1, c2, c3, c4 = st.columns([3, 1, 3, 1])
    # Display the vote buttons

    vote_disabled = (
        True
        if all([
            st.session_state.winner_selected,
        ])
        else False
    )
    with c1:
        left_button_clicked = st.button(
            "👍 Vote 1st Model",
            disabled=vote_disabled,
            on_click=lambda: (setattr(st.session_state, "winner_selected", True),),
        )
        if left_button_clicked:
            helpers.Buttons.left_button_clicked(cont1, cont2)

    with c2:
        tie_button_clicked = st.button(
            "👔 Vote Tie (1:1)",
            disabled=vote_disabled,
            on_click=lambda: (setattr(st.session_state, "winner_selected", True),),
        )
        if tie_button_clicked:
            helpers.Buttons.tie_button(cont1, cont2)

    with c3:
        no_win_button_clicked = st.button(
            "❌ No Winners (0:0)",
            disabled=vote_disabled,
            on_click=lambda: (setattr(st.session_state, "winner_selected", True),),
        )
        if no_win_button_clicked:
            helpers.Buttons.no_win_button(cont1, cont2)

    with c4:
        right_button_clicked = st.button(
            "👍 Vote 2nd Model",
            disabled=vote_disabled,
            on_click=lambda: (setattr(st.session_state, "winner_selected", True),),
        )
        if right_button_clicked:
            helpers.Buttons.right_button_clicked(cont1, cont2)
            # Add custom CSS for the buttons
    history_button_clicked = st.button("Clear Histroy")
    if history_button_clicked:
        st.session_state["chat_history1"] = []
        st.session_state["chat_history2"] = []
    with st.sidebar:
        helpers.Buttons.save_button()


if __name__ == "__main__":
    asyncio.run(main())
