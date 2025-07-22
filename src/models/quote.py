from dataclasses import dataclass
from typing import Dict


@dataclass
class Quote:
    start: float
    content: str
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict:
        return {"start": str(self.start), "content": self.content, "end": str(self.end)}
