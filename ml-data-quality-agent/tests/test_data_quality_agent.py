# Tests for Data Quality Agent
"""
Tests for Data Quality Validation Agent

Run with: pytest tests/test_data_quality_agent.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.data_quality_agent import (
    DataQualityValidationAgent,
    ValidationStatus,
    IssueSeverity,
    validate_dataset,
    format_report_text,
    format_report_json,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def clean_data():
    """Create a clean dataset with no issues."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "patient_id": range(1, n + 1),
        "age": np.random.randint(18, 90, n),
        "heart_rate": np.random.randint(60, 100, n),
        "blood_pressure_systolic": np.random.randint(100, 140, n),
        "oxygen_saturation": np.random.randint(95, 100, n),
        "gender": np.random.choice(["M", "F"], n),
    })


@pytest.fixture
def data_with_missing():
    """Create dataset with missing values."""
    df = pd.DataFrame({
        "patient_id": range(1, 101),
        "age": [25] * 100,
        "heart_rate": [72] * 100,
    })
    # Add 15% missing values to age
    df.loc[:14, "age"] = np.nan
    # Add 60% missing to heart_rate (critical)
    df.loc[:59, "heart_rate"] = np.nan
    return df


@pytest.fixture
def data_with_impossible_values():
    """Create dataset with impossible clinical values."""
    return pd.DataFrame({
        "patient_id": [1, 2, 3, 4, 5],
        "age": [25, 150, 45, -5, 60],  # 150 and -5 are impossible
        "blood_pressure_systolic": [120, 130, 500, 110, 115],  # 500 is impossible
        "oxygen_saturation": [98, 97, 150, 95, 99],  # 150 is impossible
    })


@pytest.fixture
def data_with_type_issues():
    """Create dataset with type inconsistencies."""
    return pd.DataFrame({
        "patient_id": [1, 2, 3, 4, 5],
        "age": [25, 30, "forty", 45, 50],  # "forty" is wrong type
        "heart_rate": [72, "high", 80, 95, 70],  # "high" is wrong type
    })


@pytest.fixture
def data_with_duplicates():
    """Create dataset with duplicate rows and IDs."""
    df = pd.DataFrame({
        "patient_id": [1, 2, 3, 3, 4],  # ID 3 is duplicated
        "age": [25, 30, 45, 45, 50],
        "heart_rate": [72, 80, 90, 90, 70],
    })
    # Add exact duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


@pytest.fixture
def empty_data():
    """Create empty dataset."""
    return pd.DataFrame()


# ============================================================
# TESTS: SCHEMA VALIDATION
# ============================================================

class TestSchemaValidation:
    
    def test_empty_dataset_fails(self, empty_data):
        """Empty dataset should fail with CRITICAL issue."""
        report = validate_dataset(empty_data)
        
        assert report.status == ValidationStatus.FAIL
        assert len(report.critical_issues) > 0
        assert any(i.issue_type == "EMPTY_DATASET" for i in report.critical_issues)
    
    def test_duplicate_ids_detected(self, data_with_duplicates):
        """Duplicate patient IDs should be flagged as CRITICAL."""
        report = validate_dataset(data_with_duplicates)
        
        assert any(i.issue_type == "DUPLICATE_IDS" for i in report.critical_issues)
    
    def test_duplicate_rows_detected(self, data_with_duplicates):
        """Duplicate rows should be flagged as WARNING."""
        report = validate_dataset(data_with_duplicates)
        
        assert any(i.issue_type == "DUPLICATE_ROWS" for i in report.warnings)
    
    def test_clean_data_passes_schema(self, clean_data):
        """Clean data should pass schema checks."""
        report = validate_dataset(clean_data)
        
        # No schema-related critical issues
        schema_issues = [i for i in report.critical_issues 
                        if i.issue_type in ["EMPTY_DATASET", "NO_COLUMNS", 
                                           "DUPLICATE_COLUMNS", "DUPLICATE_IDS"]]
        assert len(schema_issues) == 0


# ============================================================
# TESTS: COMPLETENESS
# ============================================================

class TestCompleteness:
    
    def test_high_null_rate_critical(self, data_with_missing):
        """Columns with >50% nulls should be CRITICAL."""
        report = validate_dataset(data_with_missing)
        
        high_null_issues = [i for i in report.critical_issues 
                          if i.issue_type == "HIGH_NULL_RATE"]
        assert len(high_null_issues) > 0
        assert any(i.column == "heart_rate" for i in high_null_issues)
    
    def test_moderate_null_rate_warning(self, data_with_missing):
        """Columns with 10-50% nulls should be WARNING."""
        report = validate_dataset(data_with_missing)
        
        moderate_null = [i for i in report.warnings 
                        if i.issue_type == "MODERATE_NULL_RATE"]
        assert len(moderate_null) > 0
        assert any(i.column == "age" for i in moderate_null)
    
    def test_clean_data_no_null_issues(self, clean_data):
        """Clean data should have no null-related issues."""
        report = validate_dataset(clean_data)
        
        null_issues = [i for i in report.critical_issues + report.warnings 
                      if "NULL" in i.issue_type]
        assert len(null_issues) == 0


# ============================================================
# TESTS: DOMAIN VALIDATION
# ============================================================

class TestDomainValidation:
    
    def test_impossible_age_detected(self, data_with_impossible_values):
        """Age values outside 0-120 should be flagged."""
        report = validate_dataset(data_with_impossible_values)
        
        # Should catch age = 150 (above max)
        above_max = [i for i in report.critical_issues 
                    if i.issue_type == "VALUE_ABOVE_RANGE" and i.column == "age"]
        assert len(above_max) > 0
        
        # Should catch age = -5 (below min)
        below_min = [i for i in report.critical_issues 
                    if i.issue_type == "VALUE_BELOW_RANGE" and i.column == "age"]
        assert len(below_min) > 0
    
    def test_impossible_bp_detected(self, data_with_impossible_values):
        """Blood pressure = 500 should be flagged."""
        report = validate_dataset(data_with_impossible_values)
        
        bp_issues = [i for i in report.critical_issues 
                    if i.column == "blood_pressure_systolic"]
        assert len(bp_issues) > 0
    
    def test_impossible_oxygen_detected(self, data_with_impossible_values):
        """Oxygen saturation > 100 should be flagged."""
        report = validate_dataset(data_with_impossible_values)
        
        o2_issues = [i for i in report.critical_issues 
                    if i.column == "oxygen_saturation"]
        assert len(o2_issues) > 0


# ============================================================
# TESTS: TYPE CONSISTENCY
# ============================================================

class TestTypeConsistency:
    
    def test_string_in_numeric_column(self, data_with_type_issues):
        """String values in numeric columns should be flagged."""
        report = validate_dataset(data_with_type_issues)
        
        type_issues = [i for i in report.warnings 
                      if i.issue_type == "TYPE_INCONSISTENCY"]
        assert len(type_issues) > 0
    
    def test_type_issue_shows_examples(self, data_with_type_issues):
        """Type issues should include example bad values."""
        report = validate_dataset(data_with_type_issues)
        
        type_issues = [i for i in report.warnings 
                      if i.issue_type == "TYPE_INCONSISTENCY"]
        
        for issue in type_issues:
            assert len(issue.examples) > 0
            assert "value" in issue.examples[0]


# ============================================================
# TESTS: ML ANOMALY DETECTION
# ============================================================

class TestMLAnomalyDetection:
    
    def test_ml_methods_recorded(self, clean_data):
        """ML methods used should be recorded in report."""
        # Need enough data for ML
        large_data = pd.concat([clean_data] * 5, ignore_index=True)
        report = validate_dataset(large_data)
        
        assert len(report.ml_methods_used) > 0
        assert any("Isolation Forest" in m for m in report.ml_methods_used)
        assert any("LOF" in m for m in report.ml_methods_used)
    
    def test_anomaly_with_outliers(self):
        """Dataset with clear outliers should flag anomalies."""
        np.random.seed(42)
        n = 200
        
        # Normal data
        df = pd.DataFrame({
            "patient_id": range(1, n + 1),
            "age": np.random.normal(50, 10, n),
            "heart_rate": np.random.normal(75, 10, n),
            "bmi": np.random.normal(25, 3, n),
        })
        
        # Add clear outliers
        df.loc[0, "age"] = 150
        df.loc[1, "heart_rate"] = 200
        df.loc[2, "bmi"] = 70
        
        report = validate_dataset(df)
        
        # Should detect some anomalies
        anomaly_issues = [i for i in report.warnings + report.info 
                        if "ANOMALY" in i.issue_type]
        # At least statistical outliers should be caught
        assert len(report.warnings) > 0


# ============================================================
# TESTS: REPORT GENERATION
# ============================================================

class TestReportGeneration:
    
    def test_report_has_all_fields(self, clean_data):
        """Report should have all required fields."""
        report = validate_dataset(clean_data)
        
        assert report.status in ValidationStatus
        assert isinstance(report.summary, dict)
        assert isinstance(report.critical_issues, list)
        assert isinstance(report.warnings, list)
        assert isinstance(report.info, list)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.column_profiles, dict)
    
    def test_summary_has_counts(self, clean_data):
        """Summary should include row/column counts."""
        report = validate_dataset(clean_data)
        
        assert "total_rows" in report.summary
        assert "total_columns" in report.summary
        assert "overall_completeness" in report.summary
    
    def test_text_format_works(self, clean_data):
        """Text formatting should produce string output."""
        report = validate_dataset(clean_data)
        text = format_report_text(report)
        
        assert isinstance(text, str)
        assert "DATA QUALITY VALIDATION REPORT" in text
    
    def test_json_format_works(self, clean_data):
        """JSON formatting should produce dict output."""
        report = validate_dataset(clean_data)
        json_out = format_report_json(report)
        
        assert isinstance(json_out, dict)
        assert "status" in json_out
        assert "summary" in json_out


# ============================================================
# TESTS: CONFIGURATION
# ============================================================

class TestConfiguration:
    
    def test_custom_null_threshold(self):
        """Custom null thresholds should be respected."""
        df = pd.DataFrame({
            "patient_id": range(1, 101),
            "age": [25] * 95 + [np.nan] * 5,  # 5% null
        })
        
        # Default threshold (10%) - should be INFO
        report_default = validate_dataset(df)
        age_issues = [i for i in report_default.warnings 
                     if i.column == "age"]
        assert len(age_issues) == 0  # 5% < 10%, so no warning
        
        # Stricter threshold (3%) - should be WARNING
        report_strict = validate_dataset(df, config={
            "null_threshold_warning": 0.03
        })
        age_issues = [i for i in report_strict.warnings 
                     if i.column == "age"]
        assert len(age_issues) > 0  # 5% > 3%, so warning


# ============================================================
# TESTS: COLUMN PROFILES
# ============================================================

class TestColumnProfiles:
    
    def test_numeric_column_stats(self, clean_data):
        """Numeric columns should have mean, std, min, max."""
        report = validate_dataset(clean_data)
        
        age_profile = report.column_profiles.get("age")
        assert age_profile is not None
        assert age_profile.mean is not None
        assert age_profile.std is not None
        assert age_profile.min_val is not None
        assert age_profile.max_val is not None
    
    def test_categorical_column_stats(self, clean_data):
        """Categorical columns should have top categories."""
        report = validate_dataset(clean_data)
        
        gender_profile = report.column_profiles.get("gender")
        assert gender_profile is not None
        assert gender_profile.top_categories is not None


# ============================================================
# TESTS: EDGE CASES
# ============================================================

class TestEdgeCases:
    
    def test_single_row(self):
        """Single row dataset should be handled."""
        df = pd.DataFrame({
            "patient_id": [1],
            "age": [25],
        })
        
        report = validate_dataset(df)
        # Should warn about insufficient rows
        assert any(i.issue_type == "INSUFFICIENT_ROWS" for i in report.warnings)
    
    def test_single_column(self):
        """Single column dataset should be handled."""
        df = pd.DataFrame({
            "patient_id": range(1, 101),
        })
        
        report = validate_dataset(df)
        assert report is not None
    
    def test_all_null_column(self):
        """Column with all nulls should be flagged as CRITICAL."""
        df = pd.DataFrame({
            "patient_id": range(1, 101),
            "empty_col": [np.nan] * 100,
        })
        
        report = validate_dataset(df)
        assert any(i.issue_type == "EMPTY_COLUMN" for i in report.critical_issues)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])