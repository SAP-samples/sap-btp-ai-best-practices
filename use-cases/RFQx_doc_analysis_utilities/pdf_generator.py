"""
PDF Generation Module for RFQ Analysis Reports

This module provides functionality to convert markdown analysis reports 
into professionally formatted PDF documents.
"""

import io
import re
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY


class MarkdownToPDFConverter:
    """
    Converts markdown content to professionally formatted PDF documents.
    """
    
    def __init__(self, project_name: Optional[str] = None):
        """
        Initialize the PDF converter.
        
        Args:
            project_name: Optional project name to include in the document header
        """
        self.project_name = project_name or "RFQ Analysis"
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the PDF."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1f4e79')
        ))
        
        # Header styles for different levels
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#2f5f8f'),
            keepWithNext=True
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.HexColor('#3f6f9f'),
            keepWithNext=True
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading3',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.HexColor('#4f7faf'),
            keepWithNext=True
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            leading=12
        ))
        
        # Bullet point style
        self.styles.add(ParagraphStyle(
            name='CustomBullet',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=4,
            leading=12
        ))
        
        # Code/monospace style
        self.styles.add(ParagraphStyle(
            name='CustomCode',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName='Courier',
            leftIndent=20,
            rightIndent=20,
            backColor=colors.HexColor('#f5f5f5'),
            borderColor=colors.HexColor('#cccccc'),
            borderWidth=1,
            borderPadding=5,
            spaceAfter=8
        ))
    
    def convert_to_pdf(self, markdown_content: str, filename: Optional[str] = None) -> io.BytesIO:
        """
        Convert markdown content to PDF.
        
        Args:
            markdown_content: The markdown content to convert
            filename: Optional filename for internal reference
            
        Returns:
            BytesIO buffer containing the PDF data
        """
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build document content
        story = []
        
        # Convert markdown to PDF elements directly (no title page)
        story.extend(self._parse_markdown_to_elements(markdown_content))
        
        # Build PDF
        doc.build(story)
        
        # Return buffer
        buffer.seek(0)
        return buffer
    
    
    def _parse_markdown_to_elements(self, markdown_content: str) -> list:
        """
        Parse markdown content and convert to ReportLab elements.
        
        Args:
            markdown_content: Raw markdown content
            
        Returns:
            List of ReportLab flowable elements
        """
        elements = []
        
        # Split content by lines for processing
        lines = markdown_content.split('\n')
        current_list_items = []
        in_code_block = False
        code_block_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:  # Empty line
                if current_list_items:
                    elements.extend(self._create_list_elements(current_list_items))
                    current_list_items = []
                if in_code_block and code_block_content:
                    elements.append(self._create_code_block(code_block_content))
                    code_block_content = []
                    in_code_block = False
                elements.append(Spacer(1, 6))
                i += 1
                continue
            
            # Handle code blocks
            if line.startswith('```'):
                if in_code_block:
                    # End of code block
                    if code_block_content:
                        elements.append(self._create_code_block(code_block_content))
                        code_block_content = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
                i += 1
                continue
            
            if in_code_block:
                code_block_content.append(lines[i])  # Preserve original formatting
                i += 1
                continue
            
            # Handle headers
            if line.startswith('#'):
                if current_list_items:
                    elements.extend(self._create_list_elements(current_list_items))
                    current_list_items = []
                
                header_level = len(line) - len(line.lstrip('#'))
                header_text = line.lstrip('#').strip()
                
                if header_level == 1:
                    style = 'CustomHeading1'
                elif header_level == 2:
                    style = 'CustomHeading2'
                else:
                    style = 'CustomHeading3'
                
                elements.append(Paragraph(header_text, self.styles[style]))
                
            # Handle horizontal rules (---, ----, etc.)
            elif re.match(r'^-{3,}$', line.strip()):
                if current_list_items:
                    elements.extend(self._create_list_elements(current_list_items))
                    current_list_items = []
                # Add horizontal rule as a spacer with a line
                elements.append(Spacer(1, 12))
                
            # Handle bullet points (only when followed by space and content)
            elif re.match(r'^[-*]\s+\w', line):
                bullet_text = line[1:].strip()
                current_list_items.append(bullet_text)
                
            # Handle numbered lists
            elif re.match(r'^\d+\.', line):
                # Extract the text after the number and period
                numbered_text = re.sub(r'^\d+\.\s*', '', line)
                current_list_items.append(numbered_text)
                
            # Handle tables (basic support)
            elif '|' in line and not line.startswith('|---'):
                if current_list_items:
                    elements.extend(self._create_list_elements(current_list_items))
                    current_list_items = []
                
                # Look ahead to collect all table rows
                table_rows = [line]
                j = i + 1
                while j < len(lines) and '|' in lines[j] and lines[j].strip():
                    if not lines[j].startswith('|---'):  # Skip separator rows
                        table_rows.append(lines[j])
                    j += 1
                
                if len(table_rows) > 1:  # Only create table if multiple rows
                    elements.append(self._create_table(table_rows))
                    i = j - 1  # Update counter to skip processed rows
                
            # Handle regular paragraphs
            else:
                if current_list_items:
                    elements.extend(self._create_list_elements(current_list_items))
                    current_list_items = []
                
                # Process text formatting (bold, italic)
                formatted_line = self._format_text(line)
                elements.append(Paragraph(formatted_line, self.styles['CustomBody']))
            
            i += 1
        
        # Handle any remaining list items
        if current_list_items:
            elements.extend(self._create_list_elements(current_list_items))
        
        # Handle any remaining code block
        if in_code_block and code_block_content:
            elements.append(self._create_code_block(code_block_content))
        
        return elements
    
    def _create_list_elements(self, items: list) -> list:
        """Create bullet point elements from a list of items."""
        elements = []
        for item in items:
            formatted_item = self._format_text(item)
            bullet_para = Paragraph(f"• {formatted_item}", self.styles['CustomBullet'])
            elements.append(bullet_para)
        return elements
    
    def _create_code_block(self, code_lines: list) -> Paragraph:
        """Create a code block element."""
        code_text = '\n'.join(code_lines)
        # Escape HTML characters in code
        code_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return Paragraph(f"<pre>{code_text}</pre>", self.styles['CustomCode'])
    
    def _create_table(self, table_rows: list) -> Table:
        """Create a table from markdown table rows with proper column sizing."""
        # Parse table data
        table_data = []
        for row in table_rows:
            # Split by | and clean up
            cells = [cell.strip() for cell in row.split('|') if cell.strip()]
            if cells:  # Only add non-empty rows
                table_data.append(cells)
        
        if not table_data:
            return Spacer(1, 6)
        
        # Calculate optimal column widths
        num_cols = len(table_data[0]) if table_data else 0
        if num_cols == 0:
            return Spacer(1, 6)
        
        # Available width (A4 page width minus margins)
        page_width = A4[0] - 144  # 72pt margins on each side
        
        # Calculate column widths based on content length
        col_widths = []
        max_lengths = [0] * num_cols
        
        # Find maximum content length for each column
        for row in table_data:
            for i, cell in enumerate(row[:num_cols]):  # Ensure we don't exceed column count
                if i < len(max_lengths):
                    max_lengths[i] = max(max_lengths[i], len(str(cell)))
        
        # Calculate proportional widths
        total_length = sum(max_lengths)
        if total_length > 0:
            for max_len in max_lengths:
                # Minimum width of 60pt, proportional distribution of remaining space
                min_width = 60
                proportional_width = (max_len / total_length) * (page_width - (num_cols * min_width))
                col_widths.append(min_width + proportional_width)
        else:
            # Fallback to equal distribution
            col_width = page_width / num_cols
            col_widths = [col_width] * num_cols
        
        # Ensure table fits within page width
        total_width = sum(col_widths)
        if total_width > page_width:
            # Scale down proportionally
            scale_factor = page_width / total_width
            col_widths = [w * scale_factor for w in col_widths]
        
        # Wrap all cell content in Paragraphs with proper formatting
        wrapped_table_data = []
        for row in table_data:
            wrapped_row = []
            for i, cell in enumerate(row[:num_cols]):
                # Always use Paragraph for proper text control and formatting
                cell_text = str(cell).strip()
                
                # Apply markdown formatting to cell content
                formatted_cell_text = self._format_text(cell_text)
                
                # Create paragraph with word wrapping enabled
                wrapped_cell = Paragraph(
                    formatted_cell_text, 
                    self.styles['Normal']
                )
                wrapped_row.append(wrapped_cell)
            
            # Ensure row has the correct number of columns
            while len(wrapped_row) < num_cols:
                wrapped_row.append(Paragraph("", self.styles['Normal']))
            wrapped_table_data.append(wrapped_row)
        
        # Create table with calculated column widths
        table = Table(wrapped_table_data, colWidths=col_widths)
        
        # Style the table
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f7faf')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        return table
    
    def _format_text(self, text: str) -> str:
        """
        Apply basic text formatting (bold, italic) for ReportLab.
        
        Args:
            text: Raw text with markdown formatting
            
        Returns:
            Text with ReportLab HTML-like formatting
        """
        # Handle inline code first to avoid conflicts (`code`)
        text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
        
        # Handle bold text first (more specific patterns before less specific)
        # **text** (double asterisk bold)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # __text__ (double underscore bold)
        text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
        
        # Handle italic text (single patterns after double patterns)
        # *text* (single asterisk italic) - avoid conflicts with already processed bold
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', text)
        # _text_ (single underscore italic) - avoid conflicts with already processed bold
        text = re.sub(r'(?<!_)_([^_]+?)_(?!_)', r'<i>\1</i>', text)
        
        return text


def create_pdf_from_markdown(markdown_content: str, project_name: str = None, filename: str = None) -> io.BytesIO:
    """
    Convenience function to create a PDF from markdown content.
    
    Args:
        markdown_content: The markdown content to convert
        project_name: Optional project name for the document
        filename: Optional filename for internal reference
        
    Returns:
        BytesIO buffer containing the PDF data
    """
    converter = MarkdownToPDFConverter(project_name)
    return converter.convert_to_pdf(markdown_content, filename)