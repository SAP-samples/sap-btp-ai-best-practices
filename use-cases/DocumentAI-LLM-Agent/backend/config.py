"""
config.py
---------
Settings singleton for the DocAI backend.

Reads from backend/.env (via python-dotenv) with sensible defaults.
Replaces the `app.core.config` import path used in the S4 module.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # S/4HANA connection
    S4_BASE_URL: str = field(default_factory=lambda: os.getenv("S4_BASE_URL", ""))
    S4_CLIENT: str = field(default_factory=lambda: os.getenv("S4_CLIENT", "100"))
    S4_USERNAME: str = field(default_factory=lambda: os.getenv("S4_USERNAME", ""))
    S4_PASSWORD: str = field(default_factory=lambda: os.getenv("S4_PASSWORD", ""))
    S4_VERIFY: bool = field(
        default_factory=lambda: os.getenv("S4_VERIFY", "false").lower()
        not in ("false", "0", "no")
    )

    # FI Supplier Invoice posting (GL-based)
    FI_COMPANY_CODE: str = field(
        default_factory=lambda: os.getenv("FI_COMPANY_CODE", "1010")
    )
    FI_EXPENSE_GL_ACCOUNT: str = field(
        default_factory=lambda: os.getenv("FI_EXPENSE_GL_ACCOUNT", "11001000")
    )

    # FI PO-based Invoice posting (MIRO equivalent)
    FI_PO_TAX_CODE: str = field(
        default_factory=lambda: os.getenv("FI_PO_TAX_CODE", "V0")
    )


settings = Settings()
