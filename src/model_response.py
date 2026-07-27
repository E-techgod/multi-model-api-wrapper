from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ModelResponse:
    """Provider independent representation of an llm response"""

    provider= str
    model= str
    content= str

    input_tokens= int
    output_tokens= int
    total_tokens= int 

    latecy_seconds= float

    finish_reason = str | None = None
    response_id= str | None = None
    request_id= str | None = None

    raw_response= Any | None = None