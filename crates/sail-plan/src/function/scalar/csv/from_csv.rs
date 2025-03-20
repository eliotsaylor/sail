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

/// Parses a column containing a CSV string into a struct with the specified schema.
///
/// Arguments:
///   - csv_str: A column or string literal containing CSV data (one line of CSV).
///   - schema_expr: A string literal or column containing the schema in DDL format.
///   - options: An optional map of CSV parsing options. Supported options include:
///     - delimiter: The character used to separate fields (default: ',')
///     - quote: The character used for quoting (default: '"')
///     - escape: The character used for escaping (default: '\')
///     - nullValue: The string that represents null values (default: "")
///     - ignoreLeadingWhiteSpace: Whether to trim leading spaces (default: false)
///     - ignoreTrailingWhiteSpace: Whether to trim trailing spaces (default: false)
///
/// Returns:
///   - A struct value that conforms to the specified schema. Returns null for unparseable input.
pub fn from_csv(input: ScalarFunctionInput) -> PlanResult<Expr> {
    let ScalarFunctionInput { arguments, .. } = input;
    let (csv_expr, schema_expr, options_expr) = match arguments.len() {
        2 => {
            let csv = arguments[0].clone();
            let schema = arguments[1].clone();
            (csv, schema, None)
        }
        3 => {
            let csv = arguments[0].clone();
            let schema = arguments[1].clone();
            let opts = Some(arguments[2].clone());
            (csv, schema, opts)
        }
        _ => return Err(PlanError::todo("from_csv expects 2 or 3 arguments")),
    };
    if let (
        Expr::Literal(ScalarValue::Utf8(Some(csv_str))),
        Expr::Literal(ScalarValue::Utf8(Some(schema_str))),
    ) = (&csv_expr, &schema_expr)
    {
        let options_map = if let Some(opt_expr) = &options_expr {
            options::extract_options_from_expr(opt_expr)?
        } else {
            HashMap::new()
        };

        return parse_csv_with_schema(csv_str, schema_str, &options_map);
    }
    let options_expr = if let Some(opt_expr) = options_expr {
        opt_expr
    } else {
        // Empty options map
        Expr::Literal(ScalarValue::Utf8(Some(String::new())))
    };
    let udf = Arc::new(datafusion_expr::ScalarUDF::new_from_impl(FromCsvUDF));
    Ok(datafusion_expr::expr::Expr::ScalarFunction(
        datafusion_expr::expr::ScalarFunction::new_udf(
            udf,
            vec![csv_expr, schema_expr, options_expr],
        ),
    ))
}

fn parse_csv_with_schema(
    csv_str: &str,
    schema_str: &str,
    options: &HashMap<String, String>,
) -> PlanResult<Expr> {
    let struct_fields = schema::parse_schema(schema_str)?;
    let csv_mode = if parsing::contains_complex_data(csv_str) {
        parsing::CsvParseMode::Complex
    } else {
        parsing::CsvParseMode::Simple
    };
    let mut csv_values = parsing::parse_csv(csv_str, options, csv_mode)?;
    schema::process_values_for_schema(&mut csv_values, &struct_fields)?;
    let mut scalar_values = Vec::with_capacity(struct_fields.len());
    for (i, (_, field_type)) in struct_fields.iter().enumerate() {
        let value = if i < csv_values.len() && !csv_values[i].is_empty() {
            match conversion::convert_csv_value(&csv_values[i], field_type) {
                Ok(scalar) => scalar,
                Err(_) => conversion::create_null_scalar_value(field_type)?,
            }
        } else {
            conversion::create_null_scalar_value(field_type)?
        };
        scalar_values.push(value);
    }
    let struct_array = schema::create_struct_array(&struct_fields, &scalar_values)?;
    Ok(Expr::Literal(ScalarValue::Struct(Arc::new(struct_array))))
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

    fn return_type(&self, args: &[DataType]) -> Result<DataType> {
        if args.len() > 1 {
            if let DataType::Utf8 = &args[1] {
                let _meta = std::collections::HashMap::from([(
                    "dynamic_schema".to_string(),
                    "true".to_string(),
                )]);
                return Ok(DataType::Struct(Fields::from(Vec::<Field>::new())));
            }
        }
        Ok(DataType::Struct(Fields::from(Vec::<Field>::new())))
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
        match (csv_arg, schema_arg) {
            (ColumnarValue::Scalar(csv_scalar), ColumnarValue::Scalar(schema_scalar)) => {
                process_scalar_input(csv_scalar, schema_scalar, &options)
            }
            (ColumnarValue::Array(csv_array), ColumnarValue::Scalar(schema_scalar)) => {
                let schema_str = match schema_scalar {
                    ScalarValue::Utf8(Some(s)) => s.clone(),
                    _ => {
                        return Err(datafusion::common::DataFusionError::Execution(
                            "Schema must be a string".to_string(),
                        ))
                    }
                };
                let struct_fields = schema::parse_schema(&schema_str)
                    .map_err(|e| datafusion::common::DataFusionError::Execution(e.to_string()))?;
                let converted_fields: Vec<(String, &str)> = struct_fields
                    .iter()
                    .map(|(name, type_str)| (name.clone(), type_str.as_str()))
                    .collect();
                process_csv_array_with_schema(
                    csv_array.as_ref(),
                    &schema_str,
                    &options,
                    &converted_fields,
                )
            }
            (ColumnarValue::Array(csv_array), ColumnarValue::Array(schema_array)) => {
                process_csv_and_schema_arrays(csv_array.as_ref(), schema_array.as_ref(), &options)
            }
            _ => Err(datafusion::common::DataFusionError::Execution(
                "Unsupported argument types for from_csv".to_string(),
            )),
        }
    }
}

fn process_scalar_input(
    csv_scalar: &ScalarValue,
    schema_scalar: &ScalarValue,
    options: &HashMap<String, String>,
) -> Result<ColumnarValue> {
    let csv_str = match csv_scalar {
        ScalarValue::Utf8(Some(s)) => s,
        _ => {
            return Err(datafusion::common::DataFusionError::Execution(
                "CSV input must be a string".to_string(),
            ));
        }
    };
    let schema_str = match schema_scalar {
        ScalarValue::Utf8(Some(s)) => {
            if s.trim().to_uppercase().starts_with("STRUCT<") {
                s.clone()
            } else {
                format!("STRUCT<{}>", s)
            }
        }
        _ => {
            return Err(datafusion::common::DataFusionError::Execution(
                "Schema must be a string".to_string(),
            ));
        }
    };
    let struct_fields = schema::parse_schema(&schema_str)
        .map_err(|e| datafusion::common::DataFusionError::Execution(e.to_string()))?;
    let csv_mode = if parsing::contains_complex_data(csv_str) {
        parsing::CsvParseMode::Complex
    } else {
        parsing::CsvParseMode::Simple
    };
    let mut csv_values = parsing::parse_csv(csv_str, options, csv_mode)?;
    schema::process_values_for_schema(&mut csv_values, &struct_fields)?;
    let mut scalar_values = Vec::with_capacity(struct_fields.len());
    for (i, (_, field_type)) in struct_fields.iter().enumerate() {
        let value = if i < csv_values.len() && !csv_values[i].is_empty() {
            match conversion::convert_csv_value(&csv_values[i], field_type) {
                Ok(scalar) => scalar,
                Err(_) => conversion::create_null_scalar_value(field_type)?,
            }
        } else {
            conversion::create_null_scalar_value(field_type)?
        };
        scalar_values.push(value);
    }
    let struct_array = schema::create_struct_array(&struct_fields, &scalar_values)?;
    Ok(ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(
        struct_array,
    ))))
}

fn process_csv_array_with_schema(
    csv_array: &dyn Array,
    _schema_str: &str,
    options: &HashMap<String, String>,
    struct_fields: &[(String, &str)],
) -> Result<ColumnarValue> {
    let csv_strings = csv_array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::common::DataFusionError::Execution(
                "Expected CSV input to be string array".to_string(),
            )
        })?;
    let mut field_arrays = Vec::with_capacity(struct_fields.len());
    for _ in 0..struct_fields.len() {
        field_arrays.push(Vec::with_capacity(csv_strings.len()));
    }
    for i in 0..csv_strings.len() {
        if csv_strings.is_null(i) {
            for field_array in &mut field_arrays {
                field_array.push(None);
            }
            continue;
        }
        let csv_str = csv_strings.value(i);
        let csv_mode = if parsing::contains_complex_data(csv_str) {
            parsing::CsvParseMode::Complex
        } else {
            parsing::CsvParseMode::Simple
        };
        let mut csv_values = match parsing::parse_csv(csv_str, options, csv_mode) {
            Ok(vals) => vals,
            Err(_) => {
                for field_array in &mut field_arrays {
                    field_array.push(None);
                }
                continue;
            }
        };
        let converted_fields: Vec<(String, String)> = struct_fields
            .iter()
            .map(|(name, type_str)| (name.clone(), type_str.to_string()))
            .collect();
        if let Err(_) = schema::process_values_for_schema(&mut csv_values, &converted_fields) {
            for field_array in &mut field_arrays {
                field_array.push(None);
            }
            continue;
        }
        for (field_idx, (_, field_type)) in struct_fields.iter().enumerate() {
            let value = if field_idx < csv_values.len() && !csv_values[field_idx].is_empty() {
                match conversion::convert_csv_value(&csv_values[field_idx], field_type) {
                    Ok(scalar) => Some(scalar),
                    Err(_) => None,
                }
            } else {
                None
            };
            field_arrays[field_idx].push(value);
        }
    }
    let field_arrays_result =
        schema::create_arrays_from_field_scalars(struct_fields, &field_arrays, csv_strings.len())?;
    let struct_array =
        schema::create_struct_array_from_fields(struct_fields, &field_arrays_result)?;
    Ok(ColumnarValue::Array(Arc::new(struct_array)))
}

fn process_csv_and_schema_arrays(
    csv_array: &dyn Array,
    schema_array: &dyn Array,
    options: &HashMap<String, String>,
) -> Result<ColumnarValue> {
    let csv_strings = csv_array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::common::DataFusionError::Execution(
                "Expected CSV input to be string array".to_string(),
            )
        })?;
    let schema_strings = schema_array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::common::DataFusionError::Execution(
                "Expected schema input to be string array".to_string(),
            )
        })?;
    if csv_strings.len() != schema_strings.len() {
        return Err(datafusion::common::DataFusionError::Execution(
            "CSV and schema arrays must have the same length".to_string(),
        ));
    }
    for i in 0..csv_strings.len() {
        if !csv_strings.is_null(i) && !schema_strings.is_null(i) {
            let csv_str = csv_strings.value(i);
            let schema_str = schema_strings.value(i);
            return process_scalar_input(
                &ScalarValue::Utf8(Some(csv_str.to_string())),
                &ScalarValue::Utf8(Some(schema_str.to_string())),
                options,
            );
        }
    }
    Ok(ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(
        StructArray::from(vec![]),
    ))))
}

fn extract_schema_literal(dt: &DataType) -> Option<String> {
    match dt {
        DataType::Struct(fields) => {
            if fields.is_empty() {
                return Some("STRUCT<>".to_string());
            }
            let field_strs: Vec<String> = fields
                .iter()
                .map(|field| {
                    let type_str = match field.data_type() {
                        DataType::Int32 => "INT",
                        DataType::Int64 => "BIGINT",
                        DataType::Float64 => "DOUBLE",
                        DataType::Boolean => "BOOLEAN",
                        DataType::Utf8 => "STRING",
                        DataType::Date32 => "DATE",
                        DataType::Timestamp(_, _) => "TIMESTAMP",
                        _ => "STRING",
                    };
                    format!("{}: {}", field.name(), type_str)
                })
                .collect();

            Some(format!("STRUCT<{}>", field_strs.join(", ")))
        }
        DataType::Utf8 => Some("STRUCT<>".to_string()),
        _ => None,
    }
}
