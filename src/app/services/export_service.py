# src/app/services/export_service.py
"""
ExportService: Export prediction results
Supports PDF and JSON formats
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ExportService:
    """
    Service để export prediction results
    
    Formats:
        - JSON: Complete data export
        - PDF: Formatted report (requires reportlab)
    """
    
    def __init__(self, export_dir: str = None):
        """
        Initialize ExportService
        
        Args:
            export_dir: Directory để lưu exported files (absolute path)
        """
        # Use absolute path in project root
        if export_dir is None:
            # Get project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
            export_dir = os.path.join(project_root, 'exports')
        
        self.export_dir = os.path.abspath(export_dir)
        self.json_dir = os.path.join(self.export_dir, 'json')
        self.pdf_dir = os.path.join(self.export_dir, 'pdf')
        
        # Create directories
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.pdf_dir, exist_ok=True)
        
        logger.info(f"✅ ExportService initialized")
        logger.info(f"   JSON dir: {self.json_dir}")
        logger.info(f"   PDF dir: {self.pdf_dir}")
    
    def export_json(self, prediction_data: Dict, filename: Optional[str] = None) -> str:
        """
        Export prediction result as JSON
        
        Args:
            prediction_data: Dict với prediction results
            filename: Optional filename (without extension)
        
        Returns:
            Path to exported JSON file
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                verdict = prediction_data.get('verdict', 'unknown').lower()
                filename = f"prediction_{verdict}_{timestamp}"
            
            # Ensure .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = os.path.join(self.json_dir, filename)
            
            # Add export metadata
            export_data = {
                'export_info': {
                    'exported_at': datetime.now().isoformat(),
                    'format': 'json',
                    'version': '2.0.0'
                },
                'prediction': prediction_data
            }
            
            # Write JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Exported JSON: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error exporting JSON: {e}", exc_info=True)
            raise
    
    def export_pdf(self, prediction_data: Dict, filename: Optional[str] = None) -> str:
        """
        Export prediction result as PDF report
        
        Args:
            prediction_data: Dict với prediction results
            filename: Optional filename (without extension)
        
        Returns:
            Path to exported PDF file
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch, cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                Image as RLImage, HRFlowable
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            logger.error("reportlab not installed. Run: pip install reportlab")
            raise ImportError("reportlab is required for PDF export. Run: pip install reportlab")
        
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                verdict = prediction_data.get('verdict', 'unknown').lower()
                filename = f"report_{verdict}_{timestamp}"
            
            # Ensure .pdf extension
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            filepath = os.path.join(self.pdf_dir, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                filepath,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            # Styles - use custom names to avoid conflict with default styles
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='ReportTitle',
                fontSize=24,
                alignment=TA_CENTER,
                spaceAfter=30,
                textColor=colors.HexColor('#1a1a2e')
            ))
            styles.add(ParagraphStyle(
                name='ReportSubtitle',
                fontSize=14,
                alignment=TA_CENTER,
                spaceAfter=20,
                textColor=colors.HexColor('#4a4a6a')
            ))
            styles.add(ParagraphStyle(
                name='ReportSectionHeader',
                fontSize=16,
                spaceBefore=20,
                spaceAfter=10,
                textColor=colors.HexColor('#1a1a2e'),
                fontName='Helvetica-Bold'
            ))
            
            # Build content
            story = []
            
            # Title
            story.append(Paragraph("DeepFake Detection Report", styles['ReportTitle']))
            story.append(Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles['ReportSubtitle']
            ))
            story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#e0e0e0')))
            story.append(Spacer(1, 20))
            
            # Verdict Section
            verdict = prediction_data.get('verdict', 'UNKNOWN')
            confidence = prediction_data.get('confidence', 0) * 100
            
            verdict_color = colors.red if verdict == 'FAKE' else colors.green
            verdict_text = f"<font color='{verdict_color}'><b>{verdict}</b></font>"
            
            story.append(Paragraph("Analysis Result", styles['ReportSectionHeader']))
            
            result_data = [
                ['Verdict', verdict],
                ['Confidence', f"{confidence:.2f}%"],
                ['Model Used', prediction_data.get('model_used', 'N/A')],
                ['Processing Time', f"{prediction_data.get('processing_time', 0):.2f}s"]
            ]
            
            result_table = Table(result_data, colWidths=[3*inch, 3*inch])
            result_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f5')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a2e')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d0d0d0'))
            ]))
            story.append(result_table)
            story.append(Spacer(1, 20))
            
            # Probabilities Section
            probabilities = prediction_data.get('probabilities', {})
            if probabilities:
                story.append(Paragraph("Probability Distribution", styles['ReportSectionHeader']))
                
                prob_data = [
                    ['Class', 'Probability'],
                    ['FAKE', f"{probabilities.get('FAKE', 0) * 100:.2f}%"],
                    ['REAL', f"{probabilities.get('REAL', 0) * 100:.2f}%"]
                ]
                
                prob_table = Table(prob_data, colWidths=[3*inch, 3*inch])
                prob_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a6a')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d0d0d0'))
                ]))
                story.append(prob_table)
                story.append(Spacer(1, 20))
            
            # File Info Section
            file_info = prediction_data.get('file_info', {})
            if file_info:
                story.append(Paragraph("File Information", styles['ReportSectionHeader']))
                
                file_data = [
                    ['File Name', file_info.get('file_name', 'N/A')],
                    ['File Type', file_info.get('file_type', 'N/A')],
                    ['File Size', f"{file_info.get('file_size', 0) / 1024:.2f} KB"]
                ]
                
                file_table = Table(file_data, colWidths=[3*inch, 3*inch])
                file_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f5')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a2e')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d0d0d0'))
                ]))
                story.append(file_table)
                story.append(Spacer(1, 20))
            
            # Video Stats (if applicable)
            stats = prediction_data.get('stats', {})
            if stats:
                story.append(Paragraph("Video Analysis Statistics", styles['ReportSectionHeader']))
                
                stats_data = [
                    ['Total Frames', str(stats.get('total_frames', 'N/A'))],
                    ['Frames Analyzed', str(stats.get('frames_analyzed', stats.get('processed_frames', 'N/A')))],
                    ['FAKE Frames', str(stats.get('fake_count', 'N/A'))],
                    ['REAL Frames', str(stats.get('real_count', 'N/A'))],
                    ['FAKE Ratio', f"{stats.get('fake_ratio', 0) * 100:.2f}%"]
                ]
                
                stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f5')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a2e')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d0d0d0'))
                ]))
                story.append(stats_table)
            
            # Footer
            story.append(Spacer(1, 40))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
            story.append(Spacer(1, 10))
            story.append(Paragraph(
                "Generated by DeepFake Detection Web App V2.0",
                ParagraphStyle(
                    name='Footer',
                    fontSize=10,
                    alignment=TA_CENTER,
                    textColor=colors.HexColor('#888888')
                )
            ))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"✅ Exported PDF: {filepath}")
            return filepath
            
        except ImportError:
            raise
        except Exception as e:
            logger.error(f"❌ Error exporting PDF: {e}", exc_info=True)
            raise
    
    def export(self, prediction_data: Dict, format: str = 'json', filename: Optional[str] = None) -> str:
        """
        Export prediction result
        
        Args:
            prediction_data: Dict với prediction results
            format: 'json' or 'pdf'
            filename: Optional filename (without extension)
        
        Returns:
            Path to exported file
        """
        if format == 'json':
            return self.export_json(prediction_data, filename)
        elif format == 'pdf':
            return self.export_pdf(prediction_data, filename)
        else:
            raise ValueError(f"Invalid format: {format}. Use 'json' or 'pdf'")
