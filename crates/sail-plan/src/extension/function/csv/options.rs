use std::collections::HashMap;

use datafusion::arrow::array::Array;
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::{lit, Expr};
use datafusion::physical_plan::ColumnarValue;

use crate::error::PlanResult;

#[derive(Debug, Clone, Default)]
pub struct CsvOptions {
    pub delimiter: char,
    pub quote: char,
    pub escape: char,
    pub null_value: String,
    pub ignore_leading_whitespace: bool,
    pub ignore_trailing_whitespace: bool,
}

impl CsvOptions {
    pub fn new() -> Self {
        Self {
            delimiter: ',',
            quote: '"',
            escape: '\\',
            null_value: String::new(),
            ignore_leading_whitespace: false,
            ignore_trailing_whitespace: false,
        }
    }
    pub fn from_hashmap(options: &HashMap<String, String>) -> Self {
        let mut csv_options = Self::new();
        if let Some(delimiter) = options.get("delimiter")
            .or_else(|| options.get("sep"))
            .and_then(|s| s.chars().next())
        {
            csv_options.delimiter = delimiter;
        }
        if let Some(quote) = options.get("quote").and_then(|s| s.chars().next()) {
            csv_options.quote = quote;
        }
        if let Some(escape) = options.get("escape").and_then(|s| s.chars().next()) {
            csv_options.escape = escape;
        }
        if let Some(null_value) = options.get("nullValue") {
            csv_options.null_value = null_value.clone();
        }
        if let Some(ignore_leading) = options.get("ignoreLeadingWhiteSpace") {
            csv_options.ignore_leading_whitespace = ignore_leading.eq_ignore_ascii_case("true");
        }
        if let Some(ignore_trailing) = options.get("ignoreTrailingWhiteSpace") {
            csv_options.ignore_trailing_whitespace = ignore_trailing.eq_ignore_ascii_case("true");
        }
        csv_options
    }

    #[allow(dead_code)]
    pub fn to_hashmap(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("delimiter".to_string(), self.delimiter.to_string());
        map.insert("quote".to_string(), self.quote.to_string());
        map.insert("escape".to_string(), self.escape.to_string());
        map.insert("nullValue".to_string(), self.null_value.clone());
        map.insert(
            "ignoreLeadingWhiteSpace".to_string(),
            self.ignore_leading_whitespace.to_string(),
        );
        map.insert(
            "ignoreTrailingWhiteSpace".to_string(),
            self.ignore_trailing_whitespace.to_string(),
        );
        map
    }
}

pub fn extract_options_from_expr(expr: &Expr) -> PlanResult<HashMap<String, String>> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(opts_str))) => {
            if opts_str.starts_with('{') && opts_str.ends_with('}') {
                let mut options = HashMap::new();
                simple_parse_json_like_string(opts_str, &mut options);
                Ok(options)
            } else {
                let mut options = HashMap::new();
                parse_options_string(opts_str, &mut options)?;
                Ok(options)
            }
        }
        Expr::ScalarFunction(scalar_function) => {
            let mut options = HashMap::new();
            let args = &scalar_function.args;
            for i in (0..args.len()).step_by(2) {
                if i + 1 < args.len() {
                    if let (
                        Expr::Literal(ScalarValue::Utf8(Some(key))),
                        Expr::Literal(ScalarValue::Utf8(Some(value))),
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

pub fn extract_options_from_columnar(arg: &ColumnarValue) -> Result<HashMap<String, String>> {
    let mut options = HashMap::new();
    match arg {
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(opts_str))) => {
            parse_options_string(opts_str, &mut options)?;
        }
        ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => {
            if struct_array.column_names().len() >= 2 {
                use datafusion::arrow::array::StringArray;
                let key_column = struct_array.column(0);
                let value_column = struct_array.column(1);
                if let (Some(key_arr), Some(value_arr)) = (
                    key_column.as_any().downcast_ref::<StringArray>(),
                    value_column.as_any().downcast_ref::<StringArray>(),
                ) {
                    if !key_arr.is_null(0) && !value_arr.is_null(0) {
                        options
                            .insert(key_arr.value(0).to_string(), value_arr.value(0).to_string());
                    }
                }
            }
        }
        ColumnarValue::Array(array) => {
            use datafusion::arrow::array::StringArray;
            if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
                if string_array.len() > 0 && !string_array.is_null(0) {
                    let opts_str = string_array.value(0);
                    parse_options_string(opts_str, &mut options)?;
                }
            }
        }
        _ => {}
    }
    Ok(options)
}

#[allow(dead_code)]
pub fn options_to_expr(options: &HashMap<String, String>) -> PlanResult<Expr> {
    let options_str = options
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(",");

    Ok(lit(ScalarValue::Utf8(Some(options_str))))
}

pub fn parse_options_string(opts_str: &str, options: &mut HashMap<String, String>) -> Result<()> {
    for part in opts_str.split(',') {
        if let Some((key, value)) = part.split_once('=') {
            options.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    Ok(())
}

pub fn simple_parse_json_like_string(s: &str, options: &mut HashMap<String, String>) {
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
        if c == '"' && (i == 0 || s.chars().nth(i - 1) != Some('\\')) {
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
