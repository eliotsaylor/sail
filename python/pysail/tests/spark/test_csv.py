import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
import datetime
import java.sql.Timestamp
import java.sql.Date
from decimal import Decimal

# Test functions for Sail's CSV functions corresponding to Spark's CsvFunctionsSuite


def test_from_csv_with_empty_options(sail):
    df = sail.createDataFrame(["1"], schema="value STRING")
    schema = "a INT"

    result = df.select(sail.from_csv("value", schema, {})).toPandas()
    expected = pd.DataFrame({"from_csv(value, a INT, {})": [{"a": 1}]})
    assert_frame_equal(result, expected)


def test_from_csv_with_non_struct_schema(sail):
    df = sail.createDataFrame(["1"], schema="value STRING")
    
    with pytest.raises(Exception) as excinfo:
        df.select(sail.from_csv("value", "ARRAY<INT>", {})).collect()
    
    assert "non-struct type" in str(excinfo.value).lower() or "invalid schema" in str(excinfo.value).lower()
    
    with pytest.raises(Exception) as excinfo:
        sail.sql("SELECT from_csv('1', 'ARRAY<INT>')").collect()
    
    assert "non-struct type" in str(excinfo.value).lower() or "invalid schema" in str(excinfo.value).lower()


def test_from_csv_with_timestamp_format(sail):
    df = sail.createDataFrame(["26/08/2015 18:00"], schema="value STRING")
    schema = "time TIMESTAMP"
    options = {"timestampFormat": "dd/MM/yyyy HH:mm"}

    result = df.select(sail.from_csv("value", schema, options)).toPandas()
    expected_timestamp = datetime.datetime(2015, 8, 26, 18, 0, 0)
    
    # Extract the timestamp value from the result
    actual_value = result.iloc[0, 0]["time"]
    assert actual_value == expected_timestamp


def test_from_csv_columnNameOfCorruptRecord(sail):
    columnNameOfCorruptRecord = "_unparsed"
    df = sail.createDataFrame(["0,2013-111-11 12:13:14", "1,1983-08-04"], schema="value STRING")
    schema = f"a INT, b DATE, {columnNameOfCorruptRecord} STRING"
    options = {"mode": "PERMISSIVE", "columnNameOfCorruptRecord": columnNameOfCorruptRecord}
    
    result = df.select(sail.from_csv("value", schema, options)).toPandas()
    
    # Check for corrupt record handling
    first_row = result.iloc[0, 0]
    second_row = result.iloc[1, 0]
    
    assert first_row["a"] == 0
    assert first_row["b"] is None
    assert first_row[columnNameOfCorruptRecord] == "0,2013-111-11 12:13:14"
    
    assert second_row["a"] == 1
    assert second_row["b"] == datetime.date(1983, 8, 4)
    assert second_row[columnNameOfCorruptRecord] is None


def test_from_csv_with_escape(sail):
    df = sail.createDataFrame(["\"#\"\""], schema="value STRING")
    schema = "str STRING"
    options = {"escape": "#"}

    result = df.select(sail.from_csv("value", schema, options)).toPandas()
    assert result.iloc[0, 0]["str"] == "\""


def test_from_csv_with_comment(sail):
    df = sail.createDataFrame(["# This line is commented"], schema="value STRING")
    schema = "str STRING"
    options = {"comment": "#"}

    result = df.select(sail.from_csv("value", schema, options)).toPandas()
    assert result.iloc[0, 0]["str"] is None


def test_from_csv_with_ignoreLeadingWhiteSpace(sail):
    df = sail.createDataFrame([" a   "], schema="value STRING")
    schema = "str STRING"
    options = {"ignoreLeadingWhiteSpace": "true"}

    result = df.select(sail.from_csv("value", schema, options)).toPandas()
    assert result.iloc[0, 0]["str"] == "a   "


def test_from_csv_with_ignoreTrailingWhiteSpace(sail):
    df = sail.createDataFrame([" a   "], schema="value STRING")
    schema = "str STRING"
    options = {"ignoreTrailingWhiteSpace": "true"}

    result = df.select(sail.from_csv("value", schema, options)).toPandas()
    assert result.iloc[0, 0]["str"] == " a"


def test_from_csv_with_nullValue(sail):
    df = sail.createDataFrame(["-"], schema="value STRING")
    schema = "str STRING"
    options = {"nullValue": "-"}

    result = df.select(sail.from_csv("value", schema, options)).toPandas()
    assert result.iloc[0, 0]["str"] is None


def test_from_csv_with_nanValue(sail):
    df = sail.createDataFrame(["#"], schema="value STRING")
    schema = "float FLOAT"
    options = {"nanValue": "#"}

    result = df.select(sail.from_csv("value", schema, options)).toPandas()
    assert pd.isna(result.iloc[0, 0]["float"])


def test_from_csv_with_positiveInf(sail):
    df = sail.createDataFrame(["#"], schema="value STRING")
    schema = "float FLOAT"
    options = {"positiveInf": "#"}

    result = df.select(sail.from_csv("value", schema, options)).toPandas()
    assert result.iloc[0, 0]["float"] == float('inf')


def test_from_csv_with_negativeInf(sail):
    df = sail.createDataFrame(["#"], schema="value STRING")
    schema = "float FLOAT"
    options = {"negativeInf": "#"}

    result = df.select(sail.from_csv("value", schema, options)).toPandas()
    assert result.iloc[0, 0]["float"] == float('-inf')


def test_from_csv_with_dateFormat(sail):
    df = sail.createDataFrame(["26/08/2015"], schema="value STRING")
    schema = "time DATE"
    options = {"dateFormat": "dd/MM/yyyy"}

    result = df.select(sail.from_csv("value", schema, options)).toPandas()
    assert result.iloc[0, 0]["time"] == datetime.date(2015, 8, 26)


def test_from_csv_with_maxCharsPerColumn(sail):
    df = sail.createDataFrame(["12345"], schema="value STRING")
    schema = "str STRING"
    options = {"maxCharsPerColumn": "2"}

    with pytest.raises(Exception) as excinfo:
        df.select(sail.from_csv("value", schema, options)).collect()
    
    assert "exceed" in str(excinfo.value).lower() or "maximum" in str(excinfo.value).lower()


def test_schema_of_csv_infers_schemas(sail):
    result = sail.sql("SELECT schema_of_csv('0.1,1')").toPandas()
    assert "STRUCT<_c0: DOUBLE, _c1: INT>" in result.iloc[0, 0]


def test_schema_of_csv_infers_schemas_with_options(sail):
    result = sail.sql("SELECT schema_of_csv('0.1 1', MAP('delimiter', ' '))").toPandas()
    assert "STRUCT<_c0: DOUBLE, _c1: INT>" in result.iloc[0, 0]


def test_to_csv_struct(sail):
    df = sail.createDataFrame([(1,)], schema="a STRUCT<x INT>")
    result = df.select(sail.to_csv("a")).toPandas()
    assert result.iloc[0, 0] == "1"


def test_to_csv_with_timestampFormat(sail):
    timestamp = datetime.datetime(2015, 8, 26, 18, 0, 0)
    df = sail.createDataFrame([(timestamp,)], schema="a STRUCT<x TIMESTAMP>")
    options = {"timestampFormat": "dd/MM/yyyy HH:mm"}

    result = df.select(sail.to_csv("a", options)).toPandas()
    assert result.iloc[0, 0] == "26/08/2015 18:00"


def test_to_csv_with_escape(sail):
    df = sail.createDataFrame([('"',)], schema="a STRUCT<x STRING>")
    options = {"escape": "#"}

    result = df.select(sail.to_csv("a", options)).toPandas()
    assert result.iloc[0, 0] == "\"#\""


def test_to_csv_with_escapeQuotes(sail):
    df = sail.createDataFrame([('test "escapeQuotes"',)], schema="a STRUCT<x STRING>")
    options = {"escapeQuotes": "false"}

    result = df.select(sail.to_csv("a", options)).toPandas()
    assert result.iloc[0, 0] == 'test "escapeQuotes"'


def test_to_csv_with_ignoreLeadingWhiteSpace(sail):
    df = sail.createDataFrame([('  a, b  , c  ',)], schema="a STRUCT<x STRING>")
    options = {"ignoreLeadingWhiteSpace": "false"}

    result = df.select(sail.to_csv("a", options)).toPandas()
    assert result.iloc[0, 0] == '"  a, b  , c"'


def test_to_csv_with_ignoreTrailingWhiteSpace(sail):
    df = sail.createDataFrame([('  a, b  , c  ',)], schema="a STRUCT<x STRING>")
    options = {"ignoreTrailingWhiteSpace": "false"}

    result = df.select(sail.to_csv("a", options)).toPandas()
    assert result.iloc[0, 0] == '"a, b  , c  "'


def test_to_csv_with_nullValue(sail):
    df = sail.createDataFrame([(None,)], schema="a STRUCT<x STRING>")
    options = {"nullValue": "-"}

    result = df.select(sail.to_csv("a", options)).toPandas()
    assert result.iloc[0, 0] == "-"


def test_to_csv_with_dateFormat(sail):
    date = datetime.date(2015, 8, 26)
    df = sail.createDataFrame([(date,)], schema="a STRUCT<x DATE>")
    options = {"dateFormat": "dd/MM/yyyy"}

    result = df.select(sail.to_csv("a", options)).toPandas()
    assert result.iloc[0, 0] == "26/08/2015"


def test_to_csv_with_emptyValue(sail):
    df = sail.createDataFrame([('',)], schema="a STRUCT<x STRING>")
    options = {"emptyValue": "-"}

    result = df.select(sail.to_csv("a", options)).toPandas()
    assert result.iloc[0, 0] == "-"


def test_from_csv_invalid_csv_check_modes(sail):
    schema = "a INT, b INT, _unparsed STRING"
    bad_rec = "\""
    df = sail.createDataFrame(["\"", "2,12"], schema="value STRING")

    # Test PERMISSIVE mode
    permissive_result = df.select(
        sail.from_csv("value", schema, {"mode": "PERMISSIVE"})
    ).toPandas()
    
    # First row should have null values and contain the bad record
    assert permissive_result.iloc[0, 0]["a"] is None
    assert permissive_result.iloc[0, 0]["b"] is None
    assert permissive_result.iloc[0, 0]["_unparsed"] == bad_rec
    
    # Second row should be parsed correctly
    assert permissive_result.iloc[1, 0]["a"] == 2
    assert permissive_result.iloc[1, 0]["b"] == 12
    assert permissive_result.iloc[1, 0]["_unparsed"] is None
    
    # Test FAILFAST mode
    with pytest.raises(Exception) as excinfo:
        df.select(sail.from_csv("value", schema, {"mode": "FAILFAST"})).collect()
    
    assert "malformed" in str(excinfo.value).lower() or "fail" in str(excinfo.value).lower()
    
    # Test DROPMALFORMED mode (Spark doesn't support it for from_csv)
    with pytest.raises(Exception) as excinfo:
        df.select(sail.from_csv("value", schema, {"mode": "DROPMALFORMED"})).collect()
    
    assert "unsupported" in str(excinfo.value).lower() or "not supported" in str(excinfo.value).lower()


def test_from_csv_uses_ddl_strings(sail):
    df = sail.createDataFrame(["""1,"haa\""""], schema="value STRING")
    result = df.select(sail.from_csv("value", "a INT, b STRING", {})).toPandas()
    
    assert result.iloc[0, 0]["a"] == 1
    assert result.iloc[0, 0]["b"] == "haa"


def test_roundtrip_to_csv_from_csv(sail):
    # Create a dataframe with a struct column
    df = sail.createDataFrame([(1,), (None,)], schema="struct STRUCT<x INT>")
    
    # Convert to CSV
    csv_df = df.select(sail.to_csv("struct").alias("csv"))
    
    # Convert back from CSV
    struct_schema = "STRUCT<x INT>"
    options = {}
    readback = csv_df.select(
        sail.from_csv("csv", struct_schema, options).alias("struct")
    )
    
    # Compare the results
    original = df.toPandas()
    result = readback.toPandas()
    
    # Check first row
    assert original.iloc[0, 0]["x"] == result.iloc[0, 0]["x"]
    
    # Check second row (None handling)
    assert original.iloc[1, 0] is None and result.iloc[1, 0] is None


def test_roundtrip_from_csv_to_csv(sail):
    # Create a dataframe with a CSV string column
    df = sail.createDataFrame(["1", None], schema="csv STRING")
    
    # Convert from CSV
    schema = "a INT"
    options = {}
    struct_df = df.select(
        sail.from_csv("csv", schema, options).alias("struct")
    )
    
    # Convert back to CSV
    readback = struct_df.select(
        sail.to_csv("struct").alias("csv")
    )
    
    # Compare the results
    original = df.toPandas()
    result = readback.toPandas()
    
    # Check first row
    assert original.iloc[0, 0] == result.iloc[0, 0]
    
    # Check second row (None handling)
    assert original.iloc[1, 0] is None and result.iloc[1, 0] is None


def test_infers_schemas_and_passes_to_from_csv(sail):
    df = sail.createDataFrame(["""0.123456789,987654321,"San Francisco\""""], schema="value STRING")
    options = {}
    
    # Use schema_of_csv to infer the schema
    result = df.select(
        sail.from_csv("value", sail.schema_of_csv("0.1,1,a"), options).alias("parsed")
    ).toPandas()
    
    # Check that the inferred schema and parsing worked correctly
    parsed = result.iloc[0, 0]
    assert isinstance(parsed["_c0"], float)
    assert parsed["_c0"] == pytest.approx(0.123456789)
    assert isinstance(parsed["_c1"], int)
    assert parsed["_c1"] == 987654321
    assert parsed["_c2"] == "San Francisco"


def test_support_to_csv_in_sql(sail):
    query = """
    SELECT to_csv(STRUCT(1 as x))
    """
    result = sail.sql(query).toPandas()
    assert result.iloc[0, 0] == "1"


def test_parse_timestamps_with_locale(sail):
    # Test various locales
    locales = ["en-US", "ko-KR", "zh-CN", "ru-RU"]
    
    for locale in locales:
        timestamp_str = "06 Nov 2018 18:00"  # This will be formatted differently based on locale
        df = sail.createDataFrame([timestamp_str], schema="value STRING")
        
        timestamp_format = "dd MMM yyyy HH:mm"
        options = {"timestampFormat": timestamp_format, "locale": locale}
        
        result = df.select(
            sail.from_csv("value", "time TIMESTAMP", options)
        ).toPandas()
        
        # The expected datetime should be the same regardless of locale formatting
        expected_timestamp = datetime.datetime(2018, 11, 6, 18, 0, 0)
        actual_timestamp = result.iloc[0, 0]["time"]
        
        # Some implementations might return java.sql.Timestamp, others might return Python datetime
        if hasattr(actual_timestamp, 'toInstant'):  # Java timestamp
            actual_timestamp = datetime.datetime.fromtimestamp(actual_timestamp.getTime() / 1000)
            
        assert actual_timestamp == expected_timestamp


def test_support_foldable_schema_by_from_csv(sail):
    df = sail.createDataFrame(["""1,"a\""""], schema="value STRING")
    options = {}
    
    # Test using a schema created by concatenating strings
    result = sail.sql("""
    SELECT from_csv('1,"a"', concat_ws(',', 'i INT', 's STRING'), {})
    """).toPandas()
    
    assert result.iloc[0, 0]["i"] == 1
    assert result.iloc[0, 0]["s"] == "a"
    
    # Test that non-foldable schemas are rejected
    with pytest.raises(Exception) as excinfo:
        sail.sql("""
        SELECT from_csv(csv, schema, {}) 
        FROM (SELECT '1' as csv, 'i INT' as schema)
        """).collect()
    
    assert "non-foldable" in str(excinfo.value).lower() or "invalid" in str(excinfo.value).lower()
    
    # Test that non-string schemas are rejected
    with pytest.raises(Exception) as excinfo:
        sail.sql("""
        SELECT from_csv('1', 1, {})
        """).collect()
    
    assert "schema" in str(excinfo.value).lower() or "invalid" in str(excinfo.value).lower()


def test_schema_of_csv_infers_foldable_csv_string(sail):
    result = sail.sql("""
    SELECT schema_of_csv(concat_ws(',', '0.1', '1'))
    """).toPandas()
    
    assert "STRUCT<_c0: DOUBLE, _c1: INT>" in result.iloc[0, 0]


def test_csv_pruning_optimization(sail):
    df = sail.createDataFrame(["a,b"], schema="csv STRING")
    
    # Test selecting a specific field from the parsed CSV
    result = df.selectExpr(
        "from_csv(csv, 'a STRING, b STRING', MAP('mode', 'failfast')).a"
    ).toPandas()
    
    assert result.iloc[0, 0] == "a"
    
    # Test selecting another field from the same parsed CSV
    result = df.selectExpr(
        "from_csv(csv, 'a STRING, b STRING', MAP('mode', 'failfast')).b"
    ).toPandas()
    
    assert result.iloc[0, 0] == "b"


def test_csv_pruning_with_corrupt_record(sail):
    df = sail.createDataFrame(["a,b,c,d"], schema="csv STRING")
    
    # Select just the corrupt record field
    result = df.selectExpr(
        "from_csv(csv, 'a STRING, b STRING, _corrupt_record STRING').a"
    ).toPandas()
    
    assert result.iloc[0, 0] == "a"


def test_from_csv_with_year_month_intervals(sail):
    # Test Year-Month interval types
    df = sail.createDataFrame(["INTERVAL '1-2' YEAR TO MONTH"], schema="value STRING")
    
    # Parse the interval string
    result = df.select(
        sail.from_csv("value", "a INTERVAL YEAR TO MONTH", {})
    ).toPandas()
    
    # Check that the interval was correctly parsed
    # Depending on Sail's implementation, the result might be different
    # In Spark it would be a Period object
    interval = result.iloc[0, 0]["a"]
    
    # Assert that the interval has the right characteristics - this might need adaptation
    # based on how Sail represents intervals
    assert "1-2" in str(interval) or "year" in str(interval).lower() and "month" in str(interval).lower()


def test_from_csv_with_day_time_intervals(sail):
    # Test Day-Time interval types
    df = sail.createDataFrame(["INTERVAL '1 02:03:04' DAY TO SECOND"], schema="value STRING")
    
    # Parse the interval string
    result = df.select(
        sail.from_csv("value", "a INTERVAL DAY TO SECOND", {})
    ).toPandas()
    
    # Check that the interval was correctly parsed
    interval = result.iloc[0, 0]["a"]
    
    # Assert that the interval has the right characteristics
    assert "1" in str(interval) and "02:03:04" in str(interval) or "day" in str(interval).lower()


def test_null_value_display_with_options(sail):
    # Test null value display with and without nullValue option
    df = sail.createDataFrame([(2, "Alice", None, "y")], schema="age LONG, name STRING, x STRING, y STRING")
    
    # Convert to struct then to CSV without nullValue option
    result1 = df.select(
        sail.to_csv("STRUCT(age, name, x, y)").alias("csv")
    ).toPandas()
    
    assert result1.iloc[0, 0] == "2,Alice,,y"
    
    # Convert with nullValue option
    options = {"nullValue": "-"}
    result2 = df.select(
        sail.to_csv("STRUCT(age, name, x, y)", options).alias("csv")
    ).toPandas()
    
    assert result2.iloc[0, 0] == "2,Alice,-,y"


def test_to_csv_with_array_type(sail):
    # Test handling of array data in to_csv
    df = sail.createDataFrame(
        [(2, "Alice", [100, 200, None, 300])], 
        schema="age LONG, name STRING, scores ARRAY<LONG>"
    )
    
    # Convert to struct then to CSV without nullValue option
    result1 = df.select(
        sail.to_csv("STRUCT(age, name, scores)").alias("csv")
    ).toPandas()
    
    # The array should be represented as a string with square brackets
    assert "[100, 200,, 300]" in result1.iloc[0, 0]
    
    # Convert with nullValue option
    options = {"nullValue": "-"}
    result2 = df.select(
        sail.to_csv("STRUCT(age, name, scores)", options).alias("csv")
    ).toPandas()
    
    # The null in the array should be replaced with the nullValue
    assert "[100, 200, -, 300]" in result2.iloc[0, 0]


def test_to_csv_with_map_type(sail):
    # Test handling of map data in to_csv
    df = sail.createDataFrame(
        [(2, "Alice", {"math": 100, "english": 200, "science": None})], 
        schema="age LONG, name STRING, scores MAP<STRING, LONG>"
    )
    
    # Convert to struct then to CSV
    result = df.select(
        sail.to_csv("STRUCT(age, name, scores)").alias("csv")
    ).toPandas()
    
    # Check that the map is represented as a string with proper formatting
    csv_str = result.iloc[0, 0]
    assert "2,Alice" in csv_str
    assert "{" in csv_str and "}" in csv_str
    assert "math -> 100" in csv_str
    assert "english -> 200" in csv_str
    assert "science ->" in csv_str


def test_to_csv_with_struct_type(sail):
    # Test handling of nested struct data in to_csv
    df = sail.createDataFrame(
        [(2, "Alice", (100, 200, None))], 
        schema="age LONG, name STRING, scores STRUCT<id1:LONG, id2:LONG, id3:LONG>"
    )
    
    # Convert to struct then to CSV
    result = df.select(
        sail.to_csv("STRUCT(age, name, scores)").alias("csv")
    ).toPandas()
    
    # Check that the nested struct is represented correctly
    csv_str = result.iloc[0, 0]
    assert "2,Alice" in csv_str
    assert "{100, 200,}" in csv_str
    
    # With nullValue option
    options = {"nullValue": "-"}
    result2 = df.select(
        sail.to_csv("STRUCT(age, name, scores)", options).alias("csv")
    ).toPandas()
    
    assert "2,Alice" in result2.iloc[0, 0]
    assert "{100, 200, -}" in result2.iloc[0, 0]