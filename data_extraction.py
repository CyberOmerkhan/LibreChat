from pydantic import BaseModel
from typing import List, Optional

class User(BaseModel):
    id: int
    name: str = "SWAT SCCS"
