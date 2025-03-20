use std::collections::HashMap;
use std::io::Cursor;

use csv::ReaderBuilder;
use datafusion::common::{DataFusionError, Result};

use super::options::CsvOptions;

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
    _options: &HashMap<String, String>,
    mode: CsvParseMode,
) -> Result<Vec<String>> {
    match mode {
        CsvParseMode::Simple => parse_csv_line(csv_str, _options),
        CsvParseMode::Complex => parse_complex_csv(csv_str, _options),
        CsvParseMode::WithoutJson => parse_csv_with_options(csv_str, _options),
        CsvParseMode::WithJson => csv_split_with_json(csv_str, _options)
            .map(|v| v.into_iter().map(String::from).collect()),
    }
}

pub fn parse_csv_line(csv_str: &str, options: &HashMap<String, String>) -> Result<Vec<String>> {
    let csv_options = CsvOptions::from_hashmap(options);
    let csv_with_newline = format!("{}\n", csv_str);
    let cursor = Cursor::new(csv_with_newline);
    let trim_mode =
        if csv_options.ignore_leading_whitespace && csv_options.ignore_trailing_whitespace {
            csv::Trim::All
        } else if csv_options.ignore_leading_whitespace {
            csv::Trim::Headers
        } else if csv_options.ignore_trailing_whitespace {
            csv::Trim::Fields
        } else {
            csv::Trim::None
        };
    let mut reader = ReaderBuilder::new()
        .delimiter(csv_options.delimiter as u8)
        .quote(csv_options.quote as u8)
        .escape(Some(csv_options.escape as u8))
        .has_headers(false)
        .trim(trim_mode)
        .flexible(true)
        .from_reader(cursor);
    let mut record = csv::StringRecord::new();
    match reader.read_record(&mut record) {
        Ok(true) => {
            let fields: Vec<String> = record.iter().map(|s| s.to_string()).collect();
            Ok(fields)
        }
        Ok(false) => Ok(Vec::new()),
        Err(e) => Err(DataFusionError::Execution(format!(
            "Error parsing CSV: {}",
            e
        ))),
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

fn parse_csv_with_options(
    csv_str: &str,
    _options: &HashMap<String, String>,
) -> Result<Vec<String>> {
    parse_csv_line(csv_str, _options)
}
