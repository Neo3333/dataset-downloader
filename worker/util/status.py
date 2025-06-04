from typing import Optional

class Status:
  def __init__(self, ok: bool, message: str = "", code: Optional[int] = None):
    self.ok = ok
    self.message = message
    self.code = code

  def is_ok(self) -> bool:
    return self.ok

  def __repr__(self):
    if self.ok:
      return "Status(OK)"
    return f"Status(ERROR, code={self.code}, message='{self.message}')"