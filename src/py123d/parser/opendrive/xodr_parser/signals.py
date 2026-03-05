from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from xml.etree.ElementTree import Element


@dataclass
class XODRSignalValidity:
    from_lane: int
    to_lane: int

    @classmethod
    def parse(cls, validity_element: Element) -> XODRSignalValidity:
        return XODRSignalValidity(
            from_lane=int(validity_element.get("fromLane")),
            to_lane=int(validity_element.get("toLane")),
        )


@dataclass
class XODRSignal:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/14_signals/14_01_introduction.html
    """

    id: int
    s: float
    t: float
    z_offset: float
    type: str
    subtype: str
    orientation: str
    dynamic: str
    # Optional
    name: Optional[str] = None
    h_offset: Optional[float] = None
    roll: Optional[float] = None
    pitch: Optional[float] = None
    country: Optional[str] = None
    value: Optional[float] = None
    text: Optional[str] = None
    height: Optional[float] = None
    width: Optional[float] = None
    validity: List[XODRSignalValidity] = field(default_factory=list)

    @classmethod
    def parse(cls, signal_element: Optional[Element]) -> XODRSignal:
        args = {}
        args["id"] = int(signal_element.get("id"))
        args["s"] = float(signal_element.get("s"))
        args["t"] = float(signal_element.get("t"))
        args["z_offset"] = float(signal_element.get("zOffset", 0.0))
        args["type"] = signal_element.get("type", "")
        args["subtype"] = signal_element.get("subtype", "")
        args["orientation"] = signal_element.get("orientation", "+")
        args["dynamic"] = signal_element.get("dynamic", "no")

        # Optional attributes
        if signal_element.get("name") is not None:
            args["name"] = signal_element.get("name")
        if signal_element.get("hOffset") is not None:
            args["h_offset"] = float(signal_element.get("hOffset"))
        if signal_element.get("roll") is not None:
            args["roll"] = float(signal_element.get("roll"))
        if signal_element.get("pitch") is not None:
            args["pitch"] = float(signal_element.get("pitch"))
        if signal_element.get("country") is not None:
            args["country"] = signal_element.get("country")
        if signal_element.get("value") is not None:
            args["value"] = float(signal_element.get("value"))
        if signal_element.get("text") is not None:
            args["text"] = signal_element.get("text")
        if signal_element.get("height") is not None:
            args["height"] = float(signal_element.get("height"))
        if signal_element.get("width") is not None:
            args["width"] = float(signal_element.get("width"))

        # Parse validity elements
        validity_list = []
        for validity_element in signal_element.findall("validity"):
            validity_list.append(XODRSignalValidity.parse(validity_element))
        args["validity"] = validity_list

        return XODRSignal(**args)


@dataclass
class XODRSignalReference:
    """Reference to existing signal (used in junction roads)"""

    id: int
    s: float
    t: float
    orientation: str
    validity: List[XODRSignalValidity] = field(default_factory=list)
    turn_relation: Optional[str] = None

    @classmethod
    def parse(cls, signal_ref_element: Element) -> XODRSignalReference:
        args = {}
        args["id"] = int(signal_ref_element.get("id"))
        args["s"] = float(signal_ref_element.get("s"))
        args["t"] = float(signal_ref_element.get("t"))
        args["orientation"] = signal_ref_element.get("orientation", "+")

        user_data = signal_ref_element.find("userData")
        if user_data is not None:
            vector_signal = user_data.find("vectorSignal")
            if vector_signal is not None and vector_signal.get("turnRelation") is not None:
                args["turn_relation"] = vector_signal.get("turnRelation")

        validity_list = []
        for validity_element in signal_ref_element.findall("validity"):
            validity_list.append(XODRSignalValidity.parse(validity_element))
        args["validity"] = validity_list

        return XODRSignalReference(**args)
