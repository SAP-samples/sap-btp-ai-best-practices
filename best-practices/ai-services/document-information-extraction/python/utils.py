import io
from IPython.display import display, IFrame
from fpdf import FPDF
from PyPDF2 import PdfReader, PdfWriter 


# Constants for overlay styling
FONT_SIZE = 10  # Font size for overlay text
MAX_TEXT_WIDTH = 150  # Maximum width for text in overlay
EXTRACTION_BOX_COLOR = (102, 255, 178)  # Color for extraction box (RGB)
EXTRACTED_VALUES_COLOR = (165, 42, 42)  # Color for extracted values (RGB)

def create_overlay(extraction_result, document_path, output_path):
    """
    Creates a PDF overlay with extracted information highlighted.

    Args:
        extraction_result (dict): The result of the document extraction process.
        document_path (str): Path to the input PDF document.
        output_path (str): Path to save the output PDF with overlay.
    """
    # Read the input PDF
    input_pdf = PdfReader(document_path)
    output_pdf = PdfWriter()

    # Initialize the PDF writer for overlay
    pdf = FPDF(unit='pt')
    pdf.set_font('Helvetica')
    pdf.set_font_size(FONT_SIZE)
    pdf.set_margins(0, 0)
    pdf.set_line_width(1)

    # Iterate through each page of the input PDF
    for n in range(len(input_pdf.pages)):
        input_page = input_pdf.pages[n]
        width = float(input_page.mediabox.width)  # Page width
        height = float(input_page.mediabox.height)  # Page height
        pdf.add_page()

        # Add legend for extraction area
        pdf.set_fill_color(*EXTRACTION_BOX_COLOR)
        pdf.rect(0.8 * width, 5, 10, 5, style='F')  # Draw extraction box
        pdf.set_xy(0.8 * width + 10, 7)  # Position text next to the box
        pdf.cell(0, txt='Extraction area', ln=1)

        # Add legend for extracted values
        pdf.set_fill_color(*EXTRACTED_VALUES_COLOR)
        pdf.rect(0.8 * width, 15, 10, 5, style='F')
        pdf.set_xy(0.8 * width + 10, 18)
        pdf.cell(0, txt='Extracted values')

        for hf in (hf for hf in extraction_result['extraction']['headerFields'] if hf['page'] == n + 1):
            x = hf['coordinates']['x']
            y = hf['coordinates']['y']
            w = hf['coordinates']['w']
            h = hf['coordinates']['h']
            # draw box around the area of the prediction
            pdf.set_draw_color(*EXTRACTION_BOX_COLOR)
            pdf.rect(x * width, y * height, w * width, h * height)

            # draw the extracted value
            pdf.set_draw_color(*EXTRACTION_BOX_COLOR)
            pdf.set_xy((x + w) * width + 2, y * height)
            pdf.multi_cell(min(pdf.get_string_width(str(hf['value'])) + 6, MAX_TEXT_WIDTH), h=FONT_SIZE,
                           txt=str(hf['value']), border=1)

        for li in (li for line in extraction_result['extraction']['lineItems'] for li in line if li['page'] == n + 1):
            x = li['coordinates']['x']
            y = li['coordinates']['y']
            w = li['coordinates']['w']
            h = li['coordinates']['h']
            # draw box around the area of the prediction
            pdf.set_draw_color(*EXTRACTION_BOX_COLOR)
            pdf.rect(x * width, y * height, w * width, h * height)

            # draw the extracted value
            pdf.set_draw_color(*EXTRACTION_BOX_COLOR)
            pdf.set_xy((x + w) * width, y * height)
            pdf.multi_cell(min(pdf.get_string_width(str(li['value'])) + 6, MAX_TEXT_WIDTH), h=FONT_SIZE,
                           txt=str(li['value']), border=1)

    # print(pdf.output())
    overlay = PdfReader(io.BytesIO(pdf.output()))

    for n in range(len(input_pdf.pages)):
        page = input_pdf.getPage(n)
        page.mergePage(overlay.getPage(n))
        output_pdf.addPage(page)

    with open(output_path, 'wb') as out:
        output_pdf.write(out)


def display_extraction(extraction_result, input_document):
    output_path = f"extraction_{extraction_result['id']}.pdf"
    create_overlay(extraction_result, input_document, output_path)
    display(IFrame(output_path, 700, 1000))


def display_capabilities(capabilities):
    document_types = capabilities['documentTypes']
    print('Available document types:', document_types)
    print('Available extraction fields:')
    for document_type in document_types:
        print(f"for '{document_type}':")
        print('\tHeader fields:')
        [print('\t\t', hf['name']) for hf in capabilities['extraction']['headerFields'] if
         (document_type in hf['supportedDocumentTypes'])]
        print('\tLine item fields:')
        [print('\t\t', li['name']) for li in capabilities['extraction']['lineItemFields'] if
         (document_type in li['supportedDocumentTypes'])]