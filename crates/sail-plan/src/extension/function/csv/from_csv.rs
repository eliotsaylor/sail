use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::{Array, StringArray, StructArray};
use datafusion::arrow::datatypes::{DataType, Field, Fields};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ColumnarValue;
use datafusion_expr::{ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};

use super::{conversion, options, parsing, schema};
use crate::error::{PlanError, PlanResult};
use crate::function::common::ScalarFunctionInput;

pub fn parse_csv_with_schema(
    csv_str: &str,
    schema_str: &str,
    options: &HashMap<String, String>,
) -> PlanResult<Expr> {
    // Normalize schema first to ensure consistent format
    let normalized_schema = normalize_schema(schema_str);

    // Debug output
    eprintln!("parse_csv_with_schema: CSV='{}', Schema='{}'", csv_str, normalized_schema);

    // Parse the schema into field definitions
    let struct_fields = match schema::parse_schema(&normalized_schema) {
        Ok(fields) => fields,
        Err(e) => return Err(PlanError::from(e)),
    };

    eprintln!("Parsed schema fields: {:?}", struct_fields);

    // Convert schema to Arrow fields (needed for metadata)
    let _arrow_fields = schema::convert_schema_to_arrow_fields(&struct_fields)?;

    // Parse CSV values using the csv crate for reliable parsing
    let csv_values = match parsing::parse_csv_line_df(csv_str, options) {
        Ok(values) => values,
        Err(e) => {
            eprintln!("Error parsing CSV: {:?}", e);
            return Err(PlanError::internal(format!("CSV parsing error: {}", e)));
        }
    };

    eprintln!("Parsed CSV values: {:?}", csv_values);

    // Ensure we have enough values to match the schema
    let mut processed_values = ensure_value_count(csv_values, struct_fields.len());

    // Apply special processing based on schema types
    if let Err(e) = schema::process_values_for_schema(&mut processed_values, &struct_fields) {
        eprintln!("Error processing values: {:?}", e);
        return Err(PlanError::invalid(format!("CSV parsing error: {}", e)));
    }

    eprintln!("Processed values: {:?}", processed_values);

    // Convert to scalar values based on schema types
    let scalar_values = match convert_to_scalar_values(&processed_values, &struct_fields) {
        Ok(values) => values,
        Err(e) => {
            eprintln!("Error converting to scalar values: {:?}", e);
            return Err(PlanError::internal(format!("Error converting values: {}", e)));
        }
    };

    eprintln!("Scalar values: {:?}", scalar_values);

    // Create the final struct array and return literal expression
    let struct_array = match schema::create_struct_array(&struct_fields, &scalar_values) {
        Ok(array) => array,
        Err(e) => {
            eprintln!("Error creating struct array: {:?}", e);
            return Err(PlanError::internal(format!("Error creating struct array: {}", e)));
        }
    };

    let struct_value = ScalarValue::Struct(Arc::new(struct_array));
    eprintln!("Final struct value: {:?}", struct_value);

    Ok(Expr::Literal(struct_value))
}

fn normalize_schema(schema_str: &str) -> String {
    let schema_str = schema_str.trim();

    // If already in STRUCT<...> format, return as is
    if schema_str.to_uppercase().starts_with("STRUCT<") {
        return schema_str.to_string();
    }

    // Parse as comma-separated field definitions
    let fields: Vec<String> = schema_str.split(',')
        .map(|field| {
            let parts: Vec<&str> = field.trim().split_whitespace().collect();
            if parts.len() >= 2 {
                let field_name = parts[0];
                let field_type = parts[1..].join(" ");
                format!("{}: {}", field_name, field_type)
            } else {
                field.trim().to_string()
            }
        })
        .collect();

    eprintln!("Normalized schema fields: {:?}", fields);
    format!("STRUCT<{}>", fields.join(", "))
}

fn ensure_value_count(values: Vec<String>, field_count: usize) -> Vec<String> {
    let mut result = values;
    eprintln!("Ensuring value count: have {}, need {}", result.len(), field_count);
    if result.len() < field_count {
        result.resize(field_count, String::new());
    }
    result
}

fn convert_to_scalar_values(
    values: &[String],
    struct_fields: &[(String, String)]
) -> Result<Vec<ScalarValue>> {
    let mut scalar_values = Vec::with_capacity(struct_fields.len());

    eprintln!("Converting to scalar values:");
    for (i, (field_name, field_type)) in struct_fields.iter().enumerate() {
        if i < values.len() && !values[i].is_empty() {
            eprintln!("  Field {}: '{}' ({}) -> converting from '{}'",
                     i, field_name, field_type, values[i]);
            match conversion::convert_csv_value(&values[i], field_type) {
                Ok(scalar) => {
                    eprintln!("    Converted to: {:?}", scalar);
                    scalar_values.push(scalar);
                },
                Err(e) => {
                    // Log the conversion error for debugging
                    eprintln!("    Error converting value '{}' to type {}: {:?}",
                             values[i], field_type, e);
                    scalar_values.push(conversion::create_null_scalar_value(field_type)?);
                }
            }
        } else {
            eprintln!("  Field {}: '{}' ({}) -> NULL (empty or missing value)",
                     i, field_name, field_type);
            scalar_values.push(conversion::create_null_scalar_value(field_type)?);
        }
    }

    Ok(scalar_values)
}

#[derive(Debug)]
pub struct FromCsvUDF;

impl ScalarUDFImpl for FromCsvUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "from_csv"
    }

    fn signature(&self) -> &Signature {
        static SIGNATURE: LazyLock<Signature> =
            LazyLock::new(|| Signature::variadic_any(Volatility::Immutable));
        &SIGNATURE
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Struct(Fields::empty()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        if args.args.len() < 2 {
            return Err(datafusion::common::DataFusionError::Execution(
                "from_csv requires at least 2 arguments: csv_string and schema".to_string(),
            ));
        }
        let csv_arg = &args.args[0];
        let schema_arg = &args.args[1];
        let options = if args.args.len() > 2 {
            options::extract_options_from_columnar(&args.args[2])?
        } else {
            HashMap::new()
        };
        let normalized_schema_arg = match schema_arg {
            ColumnarValue::Scalar(ScalarValue::Utf8(Some(schema_str))) => {
                let normalized = normalize_schema(schema_str);
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(normalized)))
            },
            other => other.clone(),
        };
        match (csv_arg, &normalized_schema_arg) {
            (ColumnarValue::Scalar(csv_scalar), ColumnarValue::Scalar(schema_scalar)) => {
                process_scalar_inputs(csv_scalar, schema_scalar, &options)
            },
            (ColumnarValue::Array(csv_array), ColumnarValue::Scalar(schema_scalar)) => {
                process_csv_array_with_scalar_schema(csv_array.as_ref(), schema_scalar, &options)
            },
            (ColumnarValue::Array(csv_array), ColumnarValue::Array(schema_array)) => {
                process_arrays(csv_array.as_ref(), schema_array.as_ref(), &options)
            },
            _ => Err(datafusion::common::DataFusionError::Execution(
                "Unsupported argument types for from_csv".to_string(),
            )),
        }
    }
}

fn process_scalar_inputs(
    csv_scalar: &ScalarValue,
    schema_scalar: &ScalarValue,
    options: &HashMap<String, String>,
) -> Result<ColumnarValue> {
    let csv_str = match csv_scalar {
        ScalarValue::Utf8(Some(s)) => s,
        ScalarValue::Utf8(None) => {
            let empty_struct = Arc::new(StructArray::from(vec![]));
            return Ok(ColumnarValue::Scalar(ScalarValue::Struct(empty_struct)));
        },
        _ => {
            return Err(datafusion::common::DataFusionError::Execution(
                "CSV input must be a string".to_string(),
            ))
        }
    };
    let schema_str = match schema_scalar {
        ScalarValue::Utf8(Some(s)) => s,
        _ => {
            return Err(datafusion::common::DataFusionError::Execution(
                "Schema must be a string".to_string(),
            ))
        }
    };
    match parse_csv_with_schema(csv_str, schema_str, options) {
        Ok(Expr::Literal(ScalarValue::Struct(struct_array))) => {
            Ok(ColumnarValue::Scalar(ScalarValue::Struct(struct_array)))
        },
        Ok(_) => {
            Err(datafusion::common::DataFusionError::Execution(
                "Unexpected result from parse_csv_with_schema".to_string(),
            ))
        },
        Err(e) => {
            Err(datafusion::common::DataFusionError::Execution(
                format!("Error parsing CSV: {}", e)
            ))
        }
    }
}

fn process_csv_array_with_scalar_schema(
    csv_array: &dyn Array,
    schema_scalar: &ScalarValue,
    options: &HashMap<String, String>,
) -> Result<ColumnarValue> {
    let schema_str = match schema_scalar {
        ScalarValue::Utf8(Some(s)) => s,
        _ => {
            return Err(datafusion::common::DataFusionError::Execution(
                "Schema must be a string".to_string(),
            ))
        }
    };
    let struct_fields = match schema::parse_schema(schema_str) {
        Ok(fields) => fields,
        Err(e) => {
            return Err(datafusion::common::DataFusionError::Execution(
                format!("Schema parsing error: {}", e)
            ))
        }
    };
    let csv_strings = match csv_array.as_any().downcast_ref::<StringArray>() {
        Some(string_array) => string_array,
        None => {
            return Err(datafusion::common::DataFusionError::Execution(
                "Expected CSV input to be string array".to_string(),
            ))
        }
    };
    let mut results = Vec::with_capacity(csv_strings.len());
    for i in 0..csv_strings.len() {
        if csv_strings.is_null(i) {
            results.push(None);
            continue;
        }
        let csv_str = csv_strings.value(i);
        match process_single_csv_row(csv_str, &struct_fields, options) {
            Ok(scalar) => results.push(Some(scalar)),
            Err(_) => results.push(None)
        }
    }
    let empty_struct = Arc::new(StructArray::from(vec![]));
    Ok(ColumnarValue::Scalar(ScalarValue::Struct(empty_struct)))
}

fn process_arrays(
    csv_array: &dyn Array,
    schema_array: &dyn Array,
    options: &HashMap<String, String>,
) -> Result<ColumnarValue> {
    let empty_struct = Arc::new(StructArray::from(vec![]));
    Ok(ColumnarValue::Scalar(ScalarValue::Struct(empty_struct)))
}

fn process_single_csv_row(
    csv_str: &str,
    struct_fields: &[(String, String)],
    options: &HashMap<String, String>,
) -> Result<ScalarValue> {
    let csv_values = match parsing::parse_csv_with_options(csv_str, options) {
        Ok(values) => values,
        Err(_) => return Err(datafusion::common::DataFusionError::Execution(
            "Failed to parse CSV".to_string()
        )),
    };
    let mut processed_values = ensure_value_count(csv_values, struct_fields.len());
    if let Err(e) = schema::process_values_for_schema(&mut processed_values, &struct_fields) {
        return Err(datafusion::common::DataFusionError::Execution(
            format!("Error processing values: {}", e)
        ));
    }
    let scalar_values = match convert_to_scalar_values(&processed_values, &struct_fields) {
        Ok(values) => values,
        Err(e) => return Err(e),
    };
    let struct_array = match schema::create_struct_array(&struct_fields, &scalar_values) {
        Ok(array) => array,
        Err(e) => return Err(e),
    };
    Ok(ScalarValue::Struct(Arc::new(struct_array)))
}