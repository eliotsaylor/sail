use std::collections::HashMap;
use std::io::Cursor;

use csv::ReaderBuilder;
use datafusion::common::Result;

use super::options::CsvOptions;
use crate::error::{PlanError, PlanResult};

pub enum CsvParseMode {
    Simple,
    Complex,
    #[allow(dead_code)]
    WithoutJson,
    #[allow(dead_code)]
    WithJson,
}

pub fn parse_csv(
    csv_str: &str,
    options: &HashMap<String, String>,
    mode: CsvParseMode,
) -> Result<Vec<String>> {
    match mode {
        CsvParseMode::Simple => parse_csv_line_df(csv_str, options),
        CsvParseMode::Complex => parse_complex_csv(csv_str, options),
        CsvParseMode::WithoutJson => parse_csv_with_options(csv_str, options),
        CsvParseMode::WithJson => {
            csv_split_with_json(csv_str, options).map(|v| v.into_iter().map(String::from).collect())
        }
    }
}

/// Parse a CSV line into fields using the CSV crate (DataFusion Result version)
///
/// This version returns a DataFusion Result for compatibility with existing code
pub fn parse_csv_line_df(csv_line: &str, options: &HashMap<String, String>) -> Result<Vec<String>> {
    parse_csv_line(csv_line, options).map_err(|e| {
        datafusion::common::DataFusionError::Execution(format!("CSV parsing error: {}", e))
    })
}

/// Parse a CSV line into fields using the CSV crate
///
/// Arguments:
///   - csv_line: A string containing a single line of CSV data
///   - options: A HashMap of CSV parsing options
///
/// Returns:
///   - A vector of extracted field values as strings
pub fn parse_csv_line(
    csv_line: &str,
    options: &HashMap<String, String>,
) -> PlanResult<Vec<String>> {
    let delimiter = options
        .get("delimiter")
        .and_then(|s| s.chars().next())
        .unwrap_or(',');
    let quote = options
        .get("quote")
        .and_then(|s| s.chars().next())
        .unwrap_or('"');
    let escape = options
        .get("escape")
        .and_then(|s| s.chars().next())
        .unwrap_or('\\');
    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter as u8)
        .quote(quote as u8)
        .escape(Some(escape as u8))
        .has_headers(false)
        .from_reader(Cursor::new(csv_line));
    if let Some(result) = reader.records().next() {
        let record =
            result.map_err(|e| PlanError::internal(format!("CSV parsing error: {}", e)))?;
        let fields: Vec<String> = record.iter().map(|s| s.to_string()).collect();
        Ok(fields)
    } else {
        Ok(Vec::new())
    }
}

pub fn parse_complex_csv(csv_str: &str, _options: &HashMap<String, String>) -> Result<Vec<String>> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;
    let mut brace_depth = 0;
    let mut bracket_depth = 0;
    for (i, c) in csv_str.char_indices() {
        match c {
            '"' if i == 0 || csv_str.chars().nth(i - 1) != Some('\\') => {
                in_quotes = !in_quotes;
            }
            '{' if !in_quotes => brace_depth += 1,
            '}' if !in_quotes => brace_depth -= 1,
            '[' if !in_quotes => bracket_depth += 1,
            ']' if !in_quotes => bracket_depth -= 1,
            ',' if !in_quotes && brace_depth == 0 && bracket_depth == 0 => {
                result.push(csv_str[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < csv_str.len() {
        result.push(csv_str[start..].trim().to_string());
    }
    Ok(result)
}

pub fn csv_split_with_json<'a>(
    csv_str: &'a str,
    _options: &HashMap<String, String>,
) -> Result<Vec<&'a str>> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;
    let mut brace_depth = 0;
    for (i, c) in csv_str.char_indices() {
        if c == '"' {
            in_quotes = !in_quotes;
        } else if !in_quotes {
            if c == '{' {
                brace_depth += 1;
            } else if c == '}' {
                brace_depth -= 1;
            } else if c == ',' && brace_depth == 0 {
                result.push(&csv_str[start..i]);
                start = i + 1;
            }
        }
    }
    if start < csv_str.len() {
        result.push(&csv_str[start..]);
    }
    Ok(result)
}

pub fn contains_complex_data(csv_str: &str) -> bool {
    csv_str.contains('{') && (csv_str.contains(':') || csv_str.contains("\""))
}

pub fn should_quote_csv_field(value: &str, delimiter: char) -> bool {
    value.contains(delimiter)
        || value.contains('"')
        || value.contains('\n')
        || value.contains('\r')
        || value.starts_with(' ')
        || value.ends_with(' ')
        || value.is_empty()
}

pub fn parse_csv_with_options(
    csv_str: &str,
    options: &HashMap<String, String>,
) -> Result<Vec<String>> {
    parse_csv_line_df(csv_str, options)
}
