import pandas as pd
from typing import Dict, Any, List, Optional


class InvoiceAPI:
    """
    In-memory Excel backend. We never write to disk; all changes are in RAM only.
    Expected columns (normalized):
      Invoice_Number, Line, PO#, IR_amount, PO_amount, GR_amount,
      Company_Code, Item_Name, User, Received_Flag
    """

    def __init__(self, excel_path: str, sheet_name: str):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self._df = self._load()

    # ---------- internal ----------
    def _load(self) -> pd.DataFrame:
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
        # normalize column names
        rename = {
            "Invoice Number": "Invoice_Number",
            "Invoice_number": "Invoice_Number",
            "invoice_number": "Invoice_Number",
            "PO Amount": "PO_amount",
            "PO_Amount": "PO_amount",
            "PO amount": "PO_amount",
            "IR Amount": "IR_amount",
            "IR_Amount": "IR_amount",
            "IR amount": "IR_amount",
            "GR Amount": "GR_amount",
            "GR_Amount": "GR_amount",
            "GR amount": "GR_amount",
            "PO number": "PO#",
            "PO Number": "PO#",
            "PO": "PO#",
            "ItemName": "Item_Name",
            "itemName": "Item_Name",
            "received_flag": "Received_Flag",
        }
        df = df.rename(columns=rename)

        # Clean up string columns - remove leading apostrophes and quotes
        string_columns = ["Invoice_Number", "PO#", "Company_Code", "Item_Name", "User"]
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lstrip("'\"").str.strip()

        # guarantee required columns
        for c in [
            "Invoice_Number",
            "Line",
            "PO#",
            "IR_amount",
            "PO_amount",
            "GR_amount",
            "Company_Code",
            "Item_Name",
            "User",
            "Received_Flag",
        ]:
            if c not in df.columns:
                df[c] = (
                    0 if c in ("Line", "IR_amount", "PO_amount", "GR_amount") else ""
                )

        # Generate Line numbers grouped by Invoice_Number
        if "Line" in df.columns and df["Line"].isna().all():
            df["Line"] = df.groupby("Invoice_Number").cumcount() + 1
        elif "Line" not in df.columns or (df["Line"] == 0).all():
            df["Line"] = df.groupby("Invoice_Number").cumcount() + 1

        # coerce numbers
        for c in ["Line", "IR_amount", "PO_amount", "GR_amount"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if "Received_Flag" in df.columns:
            df["Received_Flag"] = df["Received_Flag"].astype(bool)
        return df

    def _match_mask(self, invoice: str, line_no: Optional[int] = None):
        m = self._df["Invoice_Number"].astype(str) == str(invoice)
        if line_no is not None:
            m = m & (self._df["Line"].astype(int) == int(line_no))
        return m

    # ---------- public ----------
    def get_rows_for_user(self, user_name: str) -> List[Dict[str, Any]]:
        dfu = self._df[
            self._df["User"].astype(str).str.lower() == str(user_name).lower()
        ].copy()
        return dfu.to_dict(orient="records")

    def get_invoice_lines(self, invoice_number: str) -> Dict[str, Any]:
        rows = self._df[
            self._df["Invoice_Number"].astype(str) == str(invoice_number)
        ].copy()
        return {"rows": rows.to_dict(orient="records")}

    def update_po_amount(
        self, invoice_number: str, line_no: int, new_po_amount: float
    ) -> None:
        m = self._match_mask(invoice_number, line_no)
        self._df.loc[m, "PO_amount"] = float(new_po_amount)

    def book_gr_amount(
        self, invoice_number: str, line_no: int, new_gr_total: float
    ) -> None:
        """
        Set GR to the absolute TOTAL for that line (not delta).
        Also update Received_Flag=True if GR >= IR.
        """
        m = self._match_mask(invoice_number, line_no)
        self._df.loc[m, "GR_amount"] = float(new_gr_total)
        # mark received if fully received
        # we check per-row since mask may select multiple (shouldn't, but safe)
        rows_idx = self._df.index[m]
        for i in rows_idx:
            ir = float(self._df.at[i, "IR_amount"] or 0)
            gr = float(self._df.at[i, "GR_amount"] or 0)
            self._df.at[i, "Received_Flag"] = bool(
                gr >= ir - 1e-6
            )  # Allow small floating point differences

    def set_received_flag(
        self, invoice_number: str, line_no: int, value: bool = True
    ) -> None:
        m = self._match_mask(invoice_number, line_no)
        self._df.loc[m, "Received_Flag"] = bool(value)
