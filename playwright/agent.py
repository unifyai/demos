from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, List, Tuple
from pydantic import BaseModel, Field, create_model

from sys_msgs import INTERJECTION_TO_BROWSER_ACTION

import unify
client = unify.Unify("o3-mini@openai")
client.set_system_message(INTERJECTION_TO_BROWSER_ACTION)

SCROLLING_STATE = None


# Return Type #

class ActionName(str, Enum):
    click_button = "click_button"
    scroll = "scroll"
    start_scroll = "start_scroll"
    stop_scroll = "stop_scroll"
    new_tab = "new_tab"
    close_tab = "close_tab"
    switch_tab = "switch_tab"


class Action(BaseModel):
    action: ActionName
    # click_button
    button_idx: Optional[int] = None
    # scrolling
    direction: Optional[Literal["up", "down"]] = None
    pixels: Optional[int] = Field(None, ge=1)
    # tab ops
    tab_text: Optional[str] = None


# Structured Output #

_response_fields = {
    "rationale": (Optional[str], ...),
    "apply": (bool, ...),
}


class NewTab(BaseModel):
    rationale: Optional[str]
    apply: bool


class ScrollUp(BaseModel):
    rationale: Optional[str]
    apply: bool
    pixels: Optional[int]


class ScrollDown(BaseModel):
    rationale: Optional[str]
    apply: bool
    pixels: Optional[int]


class StartScrollingUp(BaseModel):
    rationale: Optional[str]
    apply: bool


class StartScrollingDown(BaseModel):
    rationale: Optional[str]
    apply: bool


class StopScrollingUp(BaseModel):
    rationale: Optional[str]
    apply: bool


class StopScrollingDown(BaseModel):
    rationale: Optional[str]
    apply: bool


def _construct_tab_actions(tabs: List[str], mode: str):
    return {f"{mode.lower()}_tab_" + "_".join([wrd.lower() for wrd in title.split(" ")]):
        create_model(
            f"{mode.capitalize()}Tab" + "".join([wrd.capitalize() for wrd in title.split(" ")]), **_response_fields
        )
        for title in tabs
    }


def _construct_close_tab_actions(tabs: List[str]):
    return _construct_tab_actions(tabs, "Close")


def _construct_select_tab_actions(tabs: List[str]):
    return _construct_tab_actions(tabs, "Select")


def _construct_select_button_actions(buttons: Optional[List[Tuple[int, str]]] = None):
    return {f"click_button_" + "_".join([wrd.lower() for wrd in text.split(" ")]):
        create_model(
            f"ClickButton" + "".join([wrd.capitalize() for wrd in text.split(" ")]), **_response_fields
        )
        for _, text in buttons
    }

def _construct_scroll_actions():
    if SCROLLING_STATE is None:
        return {
            "scroll_up": ScrollUp,
            "scroll_down": ScrollDown,
            "start_scrolling_up": StartScrollingUp,
            "start_scrolling_down": StartScrollingDown
        }
    elif SCROLLING_STATE == "up":
        return {
            "stop_scrolling_up": StopScrollingUp,
            "start_scrolling_down": StartScrollingDown
        }
    elif SCROLLING_STATE == "down":
        return {
            "stop_scrolling_down": StopScrollingDown,
            "start_scrolling_up": StartScrollingUp
        }
    else:
        raise Exception(f"Invalid SCROLLING_STATE {SCROLLING_STATE}")


def parse_instruction(text: str, tabs: List[str], screenshot: bytes, buttons: Optional[List[Tuple[int, str]]] = None) -> Optional[Action]:
    breakpoint()
    response_format = create_model(
        "Selection",
        new_tab=(NewTab, ...),
        **_construct_select_tab_actions(tabs),
        **_construct_close_tab_actions(tabs),
        **_construct_scroll_actions(),
        **_construct_select_button_actions(buttons)
    )
    client.set_response_format(response_format)
    ret = client.generate(text)
    breakpoint()
    return ret
