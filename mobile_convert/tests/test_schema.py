import pandas as pd
import pytest

from mobile_convert.data.schema import assert_schema


def test_schema_missing_columns():
    df = pd.DataFrame({"ID": ["a"], "Comment": ["White"]})
    with pytest.raises(ValueError):
        assert_schema(df, require_targets=True)
