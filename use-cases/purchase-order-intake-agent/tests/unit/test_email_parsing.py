"""Intake tests: e-mail parsing, inline-image filtering, Office text, upload grouping."""

import io
import struct
import unittest
import zipfile
from email.message import EmailMessage

from app.intake.documents import ROLE_DOCUMENT, ROLE_FORWARDED_EMAIL, ROLE_INLINE_IMAGE, ROLE_SOURCE_EMAIL
from app.intake.email_parsing import filter_inline_images, parse_eml
from app.intake.image_info import image_size
from app.intake.office_text import docx_to_text, xlsx_to_text
from app.intake.service import IntakeError, UploadedFile, build_messages


def png(width: int, height: int, salt: bytes = b"") -> bytes:
    """Minimal PNG header with the given size (enough for header parsing)."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00" + salt


def gif(width: int, height: int) -> bytes:
    """Minimal GIF header with the given size."""
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00\x00\x00"


def jpeg(width: int, height: int) -> bytes:
    """Minimal JPEG: SOI, APP0 segment, SOF0 segment with the given size."""
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
    sof0 = b"\xff\xc0" + struct.pack(">HBHH", 17, 8, height, width) + b"\x00" * 10
    return b"\xff\xd8" + app0 + sof0


def docx(paragraphs: list[str], table: list[list[str]]) -> bytes:
    """Build a tiny .docx with paragraphs followed by one table."""
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    para = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    rows = "".join("<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{c}</w:t></w:r></w:p></w:tc>" for c in row) + "</w:tr>"
                   for row in table)
    xml = f'<w:document xmlns:w="{w}"><w:body>{para}<w:tbl>{rows}</w:tbl></w:body></w:document>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


def xlsx(rows: list[list[object]]) -> bytes:
    """Build a tiny .xlsx workbook."""
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Order"
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def order_email(with_forward: bool = False) -> bytes:
    """Customer e-mail: HTML body with a pasted order screenshot, a logo, a duplicate and a PDF."""
    msg = EmailMessage()
    msg["From"] = "Buyer <buyer@customer.example>"
    msg["To"] = "orders@supplier.example"
    msg["Subject"] = "PO 4711"
    msg["Date"] = "Tue, 03 Mar 2026 09:03:01 +0700"
    msg.set_content("Please deliver the attached order.")
    msg.add_alternative('<p>Please deliver the <b>attached</b> order.</p><img src="cid:shot"><img src="cid:logo">',
                        subtype="html")
    html_part = msg.get_payload()[1]
    html_part.add_related(png(800, 600), maintype="image", subtype="png", cid="<shot>", filename="screenshot.png")
    html_part.add_related(gif(88, 31), maintype="image", subtype="gif", cid="<logo>", filename="logo.gif")
    html_part.add_related(png(800, 600), maintype="image", subtype="png", cid="<shot2>", filename="screenshot-copy.png")
    msg.add_attachment(b"%PDF-1.4 order", maintype="application", subtype="pdf", filename="PO-4711.pdf")
    if with_forward:
        inner = EmailMessage()
        inner["From"] = "colleague@supplier.example"
        inner["Subject"] = "Fw: price approval"
        inner.set_content("Approved, see table.")
        inner.add_alternative("<p>Approved, see table.</p>", subtype="html")
        inner.get_payload()[1].add_related(jpeg(640, 220), maintype="image", subtype="jpeg", cid="<tbl>",
                                           filename="table.jpg")
        msg.add_attachment(inner)
    return msg.as_bytes()


class ImageInfoTests(unittest.TestCase):
    """Pixel sizes from image headers."""

    def test_sizes(self):
        self.assertEqual(image_size(png(800, 600)), (800, 600))
        self.assertEqual(image_size(gif(88, 31)), (88, 31))
        self.assertEqual(image_size(jpeg(640, 220)), (640, 220))
        self.assertIsNone(image_size(b"not an image"))


class EmailParsingTests(unittest.TestCase):
    """MIME structures found on customer order e-mails."""

    def test_headers_body_and_parts(self):
        parsed = parse_eml(order_email())
        self.assertEqual(parsed.subject, "PO 4711")
        self.assertEqual(str(parsed.sent_at), "2026-03-03 02:03:01")  # converted to UTC
        self.assertIn("attached order", parsed.body_text)
        roles = [(f.role, f.file_name) for f in parsed.files]
        self.assertIn((ROLE_DOCUMENT, "PO-4711.pdf"), roles)
        self.assertEqual(sum(role == ROLE_INLINE_IMAGE for role, _ in roles), 3)

    def test_inline_images_kept_unless_logo_or_duplicate(self):
        parsed = parse_eml(order_email())
        filter_inline_images(parsed.files, min_px=150)
        by_name = {f.file_name: f for f in parsed.files}
        self.assertEqual(by_name["screenshot.png"].model_input, "NATIVE")
        self.assertEqual(by_name["screenshot.png"].model_input_reason, "800x600 px")
        self.assertEqual(by_name["logo.gif"].model_input, "NONE")
        self.assertIn("logo/icon (88x31 px)", by_name["logo.gif"].model_input_reason)
        self.assertEqual(by_name["screenshot-copy.png"].model_input, "NONE")
        self.assertIn("duplicate of screenshot.png", by_name["screenshot-copy.png"].model_input_reason)
        self.assertEqual(by_name["PO-4711.pdf"].model_input, "NATIVE")

    def test_forwarded_email_is_text_with_nested_inline_image(self):
        parsed = parse_eml(order_email(with_forward=True))
        filter_inline_images(parsed.files, min_px=150)
        forwarded = [f for f in parsed.files if f.role == ROLE_FORWARDED_EMAIL]
        self.assertEqual(len(forwarded), 1)
        self.assertEqual(forwarded[0].model_input, "TEXT")
        self.assertIn("Subject: Fw: price approval", forwarded[0].extracted_text)
        self.assertIn("Approved, see table.", forwarded[0].extracted_text)
        child = forwarded[0].children[0]
        self.assertEqual((child.role, child.model_input, child.model_input_reason), (ROLE_INLINE_IMAGE, "NATIVE", "640x220 px"))

    def test_html_only_body_is_converted(self):
        msg = EmailMessage()
        msg["Subject"] = "html"
        msg.set_content("<html><style>p{}</style><p>Line one</p><table><tr><td>A</td><td>B</td></tr></table></html>",
                        subtype="html")
        body = parse_eml(msg.as_bytes()).body_text
        self.assertIn("Line one", body)
        self.assertIn("A | B", body)
        self.assertNotIn("p{}", body)


class OfficeTextTests(unittest.TestCase):
    """Word and Excel conversion."""

    def test_docx_paragraphs_and_table(self):
        text = docx_to_text(docx(["Purchase order 99"], [["Item", "Qty"], ["Primer 4L", "12"]]))
        self.assertEqual(text.splitlines(), ["Purchase order 99", "Item | Qty", "Primer 4L | 12"])

    def test_xlsx_rows(self):
        text = xlsx_to_text(xlsx([["PO", "A-1"], [None, None], ["Primer", 12]]))
        self.assertEqual(text.splitlines(), ["## Sheet: Order", "PO | A-1", "Primer | 12"])

    def test_invalid_docx_raises(self):
        with self.assertRaises(ValueError):
            docx_to_text(b"not a zip")


class UploadGroupingTests(unittest.TestCase):
    """build_messages grouping and validation rules."""

    def test_each_eml_is_a_message_and_loose_files_share_one(self):
        files = [UploadedFile("a.eml", "message/rfc822", order_email()), UploadedFile("b.eml", None, order_email()),
                 UploadedFile("po.docx", None, docx(["PO 1"], [])), UploadedFile("scan.png", "image/png", png(900, 900))]
        messages = build_messages(files, "Deliver to site B", "SG10")
        self.assertEqual([m.source_type for m in messages], ["EML", "EML", "UPLOAD"])
        self.assertEqual(messages[0].files[0].role, ROLE_SOURCE_EMAIL)
        self.assertEqual(messages[2].body_text, "Deliver to site B")
        self.assertEqual([f.model_input for f in messages[2].files], ["TEXT", "NATIVE"])
        self.assertTrue(all(m.sales_org_hint == "SG10" for m in messages))

    def test_text_with_single_eml_is_appended(self):
        messages = build_messages([UploadedFile("a.eml", None, order_email())], "Note from sales", None)
        self.assertEqual(len(messages), 1)
        self.assertIn("Note from sales", messages[0].body_text)

    def test_text_only(self):
        messages = build_messages([], "PO 12: 4 x primer", None)
        self.assertEqual((messages[0].source_type, messages[0].subject), ("TEXT", "PO 12: 4 x primer"))

    def test_rejections(self):
        with self.assertRaises(IntakeError) as empty:
            build_messages([], "  ", None)
        self.assertEqual(empty.exception.status, 400)
        with self.assertRaises(IntakeError) as msg_file:
            build_messages([UploadedFile("mail.msg", None, b"x")], None, None)
        self.assertIn(".msg", msg_file.exception.message)
        with self.assertRaises(IntakeError):
            build_messages([UploadedFile("tool.exe", None, b"x")], None, None)


if __name__ == "__main__":
    unittest.main()
