from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, List, Tuple
from pydantic import BaseModel, Field

from sys_msgs import INTERJECTION_TO_BROWSER_ACTION

import unify
client = unify.Unify("o3-mini@openai")
client.set_system_message(INTERJECTION_TO_BROWSER_ACTION)

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


class NewTab(BaseModel):
    rationale: Optional[str]
    apply: bool


def parse_instruction(text: str, buttons: Optional[List[Tuple[int, str]]] = None, tabs: Optional[List[str]] = None, screenshot: Optional[bytes] = None) -> Optional[Action]:
    if screenshot:
        with open("img.png", "wb") as fp:
            fp.write(screenshot)
    breakpoint()
    pass
