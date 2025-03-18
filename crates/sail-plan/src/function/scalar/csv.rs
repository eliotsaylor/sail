use std::collections::HashMap;
use std::sync::Arc;

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
        1 => (arguments.one()?, std::collections::HashMap::new()),
        2 => {
            let csv = arguments[0].clone();
            let options_expr = arguments[1].clone();

            let options = match options_expr {
                expr::Expr::Literal(ScalarValue::Map(map_array)) => {
                    extract_map_options(&map_array)?
                }
                expr::Expr::Literal(ScalarValue::Utf8(Some(opts_str))) => {
                    if opts_str.starts_with('{') && opts_str.ends_with('}') {
                        let mut options = std::collections::HashMap::new();
                        simple_parse_json_like_string(&opts_str, &mut options);
                        options
                    } else {
                        parse_options_string(&opts_str)?
                    }
                }
                expr::Expr::ScalarFunction(ref scalar_function) => {
                    let mut options = std::collections::HashMap::new();
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
                    options
                }
                _ => std::collections::HashMap::new(),
            };

            (csv, options)
        }
        _ => return Err(PlanError::todo("schema_of_csv expects 1 or 2 arguments")),
    };

    if let expr::Expr::Literal(ScalarValue::Utf8(Some(csv_str))) = csv_expr {
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
    } else {
        Err(PlanError::todo(
            "schema_of_csv requires a foldable string input",
        ))
    }
}

/// Extracts options from a MapArray
fn extract_map_options(
    map_array: &Arc<datafusion::arrow::array::MapArray>,
) -> PlanResult<std::collections::HashMap<String, String>> {
    let mut options = std::collections::HashMap::new();
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
fn parse_options_string(opts_str: &str) -> PlanResult<std::collections::HashMap<String, String>> {
    let mut options = std::collections::HashMap::new();

    for part in opts_str.split(',') {
        if let Some((key, value)) = part.split_once('=') {
            options.insert(key.trim().to_string(), value.trim().to_string());
        }
    }

    Ok(options)
}

/// Parses a CSV line into fields according to CSV parsing rules
fn parse_csv_line(
    csv_str: &str,
    options: &std::collections::HashMap<String, String>,
) -> PlanResult<Vec<String>> {
    let delimiter = options
        .get("delimiter")
        .and_then(|s| s.chars().next())
        .unwrap_or(',');
    let quote = options
        .get("quote")
        .and_then(|s| s.chars().next())
        .unwrap_or('"');

    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = csv_str.chars().peekable();

    while let Some(c) = chars.next() {
        if c == quote {
            if let Some(&next_char) = chars.peek() {
                if next_char == quote {
                    field.push(quote);
                    chars.next();
                    continue;
                }
            }
            in_quotes = !in_quotes;
        } else if c == delimiter && !in_quotes {
            fields.push(std::mem::take(&mut field));
        } else {
            field.push(c);
        }
    }

    fields.push(field);
    Ok(fields)
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

/// Infers data types for CSV field values
///
/// Type inference follows this order of preference:
/// 1. INT for integer values
/// 2. DOUBLE for floating-point numbers
/// 3. BOOLEAN for true/false values (case insensitive)
/// 4. DATE/TIMESTAMP for date/time strings
/// 5. STRING as the default type
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

            if is_date_format(trimmed) {
                return "DATE".to_string();
            }

            if is_timestamp_format(trimmed) {
                return "TIMESTAMP".to_string();
            }

            "STRING".to_string()
        })
        .collect()
}

/// Checks if a string appears to be in a date format (YYYY-MM-DD)
fn is_date_format(value: &str) -> bool {
    if value.len() == 10 && value.chars().nth(4) == Some('-') && value.chars().nth(7) == Some('-') {
        let parts: Vec<&str> = value.split('-').collect();
        if parts.len() == 3 {
            return parts.iter().all(|part| part.parse::<i32>().is_ok());
        }
    }
    false
}

/// Checks if a string appears to be in a timestamp format
fn is_timestamp_format(value: &str) -> bool {
    if value.len() >= 19
        && value.chars().nth(4) == Some('-')
        && value.chars().nth(7) == Some('-')
        && value.chars().nth(10) == Some(' ')
        && value.chars().nth(13) == Some(':')
        && value.chars().nth(16) == Some(':')
    {
        return true;
    }

    if value.len() == 8 && value.chars().nth(2) == Some(':') && value.chars().nth(5) == Some(':') {
        let parts: Vec<&str> = value.split(':').collect();
        if parts.len() == 3 {
            return parts.iter().all(|part| part.parse::<i32>().is_ok());
        }
    }

    false
}

pub(super) fn list_built_in_csv_functions() -> Vec<(&'static str, ScalarFunction)> {
    use crate::function::common::ScalarFunctionBuilder as F;
    vec![
        ("from_csv", F::unknown("from_csv")),
        ("schema_of_csv", F::custom(schema_of_csv)),
        ("to_csv", F::unknown("to_csv")),
    ]
}
