import pytest

from llm_review_analysis.db.sql_validator import SQLValidationError, validate_select_sql


def test_valid_select_allowed():
    sql = validate_select_sql(
        "SELECT COUNT(*) AS n FROM sample_product",
        allowed_tables=["sample_product"],
    )
    assert sql == "SELECT COUNT(*) AS n FROM sample_product"


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM sample_product",
        "SELECT * FROM sample_product; DROP TABLE sample_product",
        "PRAGMA table_info(sample_product)",
        "SELECT secret FROM sample_product",
        "SELECT * FROM other_table",
    ],
)
def test_rejects_unsafe_or_unknown_sql(sql):
    with pytest.raises(SQLValidationError):
        validate_select_sql(sql, allowed_tables=["sample_product"])
