use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use csv::ReaderBuilder;
use datafusion::arrow::array::Array;
use datafusion_common::ScalarValue;
use datafusion_expr::{expr, lit};

use crate::error::{PlanError, PlanResult};
use crate::function::common::{ScalarFunction, ScalarFunctionInput};
use crate::utils::ItemTaker;

/// Infers the schema of a CSV string and returns it in DDL format.
///
/// Arguments:
///   - csv_str: A string literal containing CSV data. The function expects a
///     single line of CSV data.
///   - options: An optional map of CSV parsing options. Supported options include:
///     - delimiter: The character used to separate fields (default: ',')
///     - quote: The character used for quoting (default: '"')
///     - escape: The character used for escaping (default: '\')
///
/// Returns:
///   - A string literal in DDL format representing the inferred schema, in the form:
///     "STRUCT<_c0: TYPE1, _c1: TYPE2, ...>"
fn schema_of_csv(input: ScalarFunctionInput) -> PlanResult<expr::Expr> {
    let ScalarFunctionInput { arguments, .. } = input;

    let (csv_expr, options) = match arguments.len() {
        1 => (arguments.one()?, HashMap::new()),
        2 => {
            let csv = arguments[0].clone();
            let options = extract_options(&arguments[1])?;
            (csv, options)
        }
        _ => return Err(PlanError::todo("schema_of_csv expects 1 or 2 arguments")),
    };

    let csv_str = match csv_expr {
        expr::Expr::Literal(ScalarValue::Utf8(Some(csv))) => csv,
        _ => {
            return Err(PlanError::todo(
                "schema_of_csv requires a foldable string input",
            ))
        }
    };

    let fields = parse_csv_line(&csv_str, &options)?;
    let field_types = infer_field_types(&fields);

    let schema_parts: Vec<String> = fields
        .iter()
        .enumerate()
        .zip(field_types.iter())
        .map(|((i, _), field_type)| format!("_c{}: {}", i, field_type))
        .collect();

    let schema_ddl = format!("STRUCT<{}>", schema_parts.join(", "));

    Ok(lit(ScalarValue::Utf8(Some(schema_ddl))))
}

/// Parse a CSV line and return the fields as a vector of strings
fn parse_csv_line(csv_str: &str, options: &HashMap<String, String>) -> PlanResult<Vec<String>> {
    // Extract CSV options with defaults
    let delimiter = options
        .get("delimiter")
        .and_then(|s| s.chars().next())
        .unwrap_or(',') as u8;

    let quote = options
        .get("quote")
        .and_then(|s| s.chars().next())
        .unwrap_or('"') as u8;

    let escape = options
        .get("escape")
        .and_then(|s| s.chars().next())
        .unwrap_or('\\') as u8;

    let csv_with_newline = format!("{}\n", csv_str);
    let cursor = Cursor::new(csv_with_newline);

    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter)
        .quote(quote)
        .escape(Some(escape))
        .has_headers(false)
        .flexible(true)
        .from_reader(cursor);

    let mut record = csv::StringRecord::new();
    match reader.read_record(&mut record) {
        Ok(true) => {
            let fields: Vec<String> = record.iter().map(|s| s.to_string()).collect();
            Ok(fields)
        }
        Ok(false) => Ok(Vec::new()),
        Err(e) => Err(PlanError::invalid(format!("Error parsing CSV: {}", e))),
    }
}

/// Extract options from various input expressions
fn extract_options(expr: &expr::Expr) -> PlanResult<HashMap<String, String>> {
    match expr {
        expr::Expr::Literal(ScalarValue::Map(map_array)) => extract_map_options(map_array),
        expr::Expr::Literal(ScalarValue::Utf8(Some(opts_str))) => {
            if opts_str.starts_with('{') && opts_str.ends_with('}') {
                let mut options = HashMap::new();
                simple_parse_json_like_string(opts_str, &mut options);
                Ok(options)
            } else {
                parse_options_string(opts_str)
            }
        }
        expr::Expr::ScalarFunction(scalar_function) => {
            let mut options = HashMap::new();
            let args = &scalar_function.args;
            for i in (0..args.len()).step_by(2) {
                if i + 1 < args.len() {
                    if let (
                        expr::Expr::Literal(ScalarValue::Utf8(Some(key))),
                        expr::Expr::Literal(ScalarValue::Utf8(Some(value))),
                    ) = (&args[i], &args[i + 1])
                    {
                        options.insert(key.clone(), value.clone());
                    }
                }
            }
            Ok(options)
        }
        _ => Ok(HashMap::new()),
    }
}

/// Infer data types for CSV field values
fn infer_field_types(fields: &[String]) -> Vec<String> {
    fields
        .iter()
        .map(|field| {
            let trimmed = field.trim();

            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") {
                return "STRING".to_string();
            }

            if trimmed.parse::<i64>().is_ok() {
                return "INT".to_string();
            }

            if trimmed.parse::<f64>().is_ok() {
                return "DOUBLE".to_string();
            }

            if trimmed.eq_ignore_ascii_case("true") || trimmed.eq_ignore_ascii_case("false") {
                return "BOOLEAN".to_string();
            }

            if trimmed.len() == 10 && trimmed.matches('-').count() == 2 {
                if let [year, month, day] = trimmed.split('-').collect::<Vec<_>>()[..] {
                    if year.parse::<i32>().is_ok()
                        && month.parse::<i32>().is_ok()
                        && day.parse::<i32>().is_ok()
                    {
                        return "DATE".to_string();
                    }
                }
            }

            if (trimmed.len() >= 19
                && trimmed.contains(' ')
                && trimmed.matches(':').count() == 2
                && trimmed.matches('-').count() == 2)
                || (trimmed.len() == 8 && trimmed.matches(':').count() == 2)
            {
                return "TIMESTAMP".to_string();
            }

            "STRING".to_string()
        })
        .collect()
}

/// Extracts options from a MapArray
fn extract_map_options(
    map_array: &Arc<datafusion::arrow::array::MapArray>,
) -> PlanResult<HashMap<String, String>> {
    let mut options = HashMap::new();
    let map_array = map_array.as_ref();
    let keys = map_array.keys();
    let values = map_array.values();

    if let Some(key_array) = keys
        .as_any()
        .downcast_ref::<datafusion::arrow::array::StringArray>()
    {
        if let Some(value_array) = values
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StringArray>()
        {
            for i in 0..map_array.len() {
                let key = key_array.value(i);
                let value = value_array.value(i);
                options.insert(key.to_string(), value.to_string());
            }
        }
    }

    Ok(options)
}

/// Parses options from a string like "key1=value1,key2=value2"
fn parse_options_string(opts_str: &str) -> PlanResult<HashMap<String, String>> {
    let mut options = HashMap::new();

    for part in opts_str.split(',') {
        if let Some((key, value)) = part.split_once('=') {
            options.insert(key.trim().to_string(), value.trim().to_string());
        }
    }

    Ok(options)
}

/// A simple parser for JSON-like strings of the form {"key":"value","key2":"value2"}
fn simple_parse_json_like_string(s: &str, options: &mut HashMap<String, String>) {
    let s = s.trim();
    let s = if s.starts_with('{') && s.ends_with('}') {
        &s[1..s.len() - 1]
    } else {
        s
    };

    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;

    for (i, c) in s.char_indices() {
        if c == '"' {
            in_quotes = !in_quotes;
        } else if c == ',' && !in_quotes {
            parts.push(&s[start..i]);
            start = i + 1;
        }
    }

    if start < s.len() {
        parts.push(&s[start..]);
    }

    for part in parts {
        let part = part.trim();

        if let Some(colon_pos) = part.find(':') {
            let key_part = &part[0..colon_pos].trim();
            let value_part = &part[colon_pos + 1..].trim();

            let key = if key_part.starts_with('"') && key_part.ends_with('"') {
                &key_part[1..key_part.len() - 1]
            } else {
                key_part
            };

            let value = if value_part.starts_with('"') && value_part.ends_with('"') {
                &value_part[1..value_part.len() - 1]
            } else {
                value_part
            };

            options.insert(key.to_string(), value.to_string());
        }
    }
}

pub(super) fn list_built_in_csv_functions() -> Vec<(&'static str, ScalarFunction)> {
    use crate::function::common::ScalarFunctionBuilder as F;
    vec![
        ("from_csv", F::unknown("from_csv")),
        ("schema_of_csv", F::custom(schema_of_csv)),
        ("to_csv", F::unknown("to_csv")),
    ]
}
