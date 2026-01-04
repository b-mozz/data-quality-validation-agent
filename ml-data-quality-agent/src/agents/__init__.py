# Agents module
"""
Agents module for Nimblemind Case Study.

Available agents:
- DataQualityValidationAgent: Input data validation
- EthicsAndBiasCheckerAgent: Model fairness checking (coming soon)
"""

from .data_quality_agent import (
    DataQualityValidationAgent,
    ValidationStatus,
    IssueSeverity,
    ValidationReport,
    Issue,
    ColumnProfile,
    validate_dataset,
    validate_file,
    format_report_text,
    format_report_json,
)

__all__ = [
    "DataQualityValidationAgent",
    "ValidationStatus", 
    "IssueSeverity",
    "ValidationReport",
    "Issue",
    "ColumnProfile",
    "validate_dataset",
    "validate_file",
    "format_report_text",
    "format_report_json",
]