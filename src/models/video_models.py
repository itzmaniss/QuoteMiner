from dataclasses import dataclass


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    confidence: float = 0.0


@dataclass
class CropRegion:
    x: int
    y: int
    width: int
    height: int
