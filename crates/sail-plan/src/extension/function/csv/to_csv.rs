use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::{Array, StringArray, StructArray};
use datafusion::arrow::datatypes::{DataType, TimeUnit};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::{lit, Expr};
use datafusion::physical_plan::ColumnarValue;
use datafusion_expr::{ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};

use super::{conversion, options, parsing};
use crate::error::{PlanError, PlanResult};
use crate::function::common::ScalarFunctionInput;
use crate::utils::ItemTaker;

pub fn extract_struct_field_values(
    struct_array: &StructArray,
    options: &HashMap<String, String>,
) -> PlanResult<Vec<String>> {
    let mut field_values = Vec::new();

    for i in 0..struct_array.columns().len() {
        let column = struct_array.column(i);
        let field_name = struct_array.column_names()[i];
        if column.len() > 0 {
            let value_str = format_field_value(column, 0, options)?;
            field_values.push(value_str);
        } else {
            field_values.push(format!("\"{}\"", field_name));
        }
    }

    Ok(field_values)
}

fn format_field_value(
    column: &Arc<dyn Array>,
    index: usize,
    options: &HashMap<String, String>,
) -> PlanResult<String> {
    if column.is_null(index) {
        return Ok(String::new());
    }
    match column.data_type() {
        DataType::Boolean => {
            let array = column
                .as_any()
                .downcast_ref::<datafusion::arrow::array::BooleanArray>()
                .ok_or_else(|| PlanError::internal("Failed to downcast to BooleanArray"))?;
            Ok(array.value(index).to_string())
        }
        DataType::Int32 => {
            let array = column
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int32Array>()
                .ok_or_else(|| PlanError::internal("Failed to downcast to Int32Array"))?;
            Ok(array.value(index).to_string())
        }
        DataType::Int64 => {
            let array = column
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .ok_or_else(|| PlanError::internal("Failed to downcast to Int64Array"))?;
            Ok(array.value(index).to_string())
        }
        DataType::Float64 => {
            let array = column
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Float64Array>()
                .ok_or_else(|| PlanError::internal("Failed to downcast to Float64Array"))?;
            Ok(array.value(index).to_string())
        }
        DataType::Utf8 => {
            let array = column
                .as_any()
                .downcast_ref::<datafusion::arrow::array::StringArray>()
                .ok_or_else(|| PlanError::internal("Failed to downcast to StringArray"))?;
            Ok(array.value(index).to_string())
        }
        DataType::Date32 => {
            let array = column
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Date32Array>()
                .ok_or_else(|| PlanError::internal("Failed to downcast to Date32Array"))?;

            let days_since_epoch = array.value(index);
            let _format = options
                .get("dateFormat")
                .cloned()
                .unwrap_or_else(|| "%Y-%m-%d".to_string());
            conversion::format_date(days_since_epoch)
                .map_err(|e| PlanError::internal(format!("Error formatting date: {}", e)))
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let array = column
                .as_any()
                .downcast_ref::<datafusion::arrow::array::TimestampMicrosecondArray>()
                .ok_or_else(|| {
                    PlanError::internal("Failed to downcast to TimestampMicrosecondArray")
                })?;
            let micros = array.value(index);
            let _format = options
                .get("timestampFormat")
                .cloned()
                .unwrap_or_else(|| "%Y-%m-%d %H:%M:%S".to_string());
            conversion::format_timestamp(micros)
                .map_err(|e| PlanError::internal(format!("Error formatting timestamp: {}", e)))
        }
        _ => Ok(format!("{:?}", column)),
    }
}

pub fn format_as_csv(
    field_values: &[String],
    options: &HashMap<String, String>,
) -> PlanResult<String> {
    let delimiter = options
        .get("delimiter")
        .map(|s| s.chars().next().unwrap_or(','))
        .unwrap_or(',');
    let quote = options
        .get("quote")
        .map(|s| s.chars().next().unwrap_or('"'))
        .unwrap_or('"');
    let escape = options
        .get("escape")
        .map(|s| s.chars().next().unwrap_or('\\'))
        .unwrap_or('\\');
    let quoted_values: Vec<String> = field_values
        .iter()
        .map(|value| {
            if parsing::should_quote_csv_field(value, delimiter) {
                let escaped_value =
                    value.replace(quote.to_string().as_str(), &format!("{}{}", escape, quote));
                format!("{}{}{}", quote, escaped_value, quote)
            } else {
                value.clone()
            }
        })
        .collect();
    Ok(quoted_values.join(&delimiter.to_string()))
}

#[derive(Debug)]
pub struct ToCsvUDF;

impl ScalarUDFImpl for ToCsvUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "to_csv"
    }

    fn signature(&self) -> &Signature {
        static SIGNATURE: LazyLock<Signature> =
            LazyLock::new(|| Signature::variadic_any(Volatility::Immutable));
        &SIGNATURE
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        if args.args.is_empty() {
            return Err(datafusion::common::DataFusionError::Execution(
                "to_csv requires at least 1 argument: struct value".to_string(),
            ));
        }
        let struct_arg = &args.args[0];
        let options = if args.args.len() > 1 {
            options::extract_options_from_columnar(&args.args[1])?
        } else {
            HashMap::new()
        };
        match struct_arg {
            ColumnarValue::Scalar(scalar) => {
                if let ScalarValue::Struct(struct_array) = scalar {
                    struct_to_csv_scalar(struct_array, &options)
                } else {
                    Ok(ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                        scalar.to_string(),
                    ))))
                }
            }
            ColumnarValue::Array(array) => {
                if let Some(struct_array) = array.as_any().downcast_ref::<StructArray>() {
                    process_struct_array_to_csv(struct_array, &options)
                } else {
                    let len = array.len();
                    let mut string_values = Vec::with_capacity(len);
                    for i in 0..len {
                        if array.is_null(i) {
                            string_values.push(None);
                        } else {
                            let value_str = match array.data_type() {
                                DataType::Int32 => {
                                    if let Some(arr) = array
                                        .as_any()
                                        .downcast_ref::<datafusion::arrow::array::Int32Array>(
                                    ) {
                                        Some(arr.value(i).to_string())
                                    } else {
                                        Some(format!("row_{}", i))
                                    }
                                }
                                DataType::Int64 => {
                                    if let Some(arr) = array
                                        .as_any()
                                        .downcast_ref::<datafusion::arrow::array::Int64Array>(
                                    ) {
                                        Some(arr.value(i).to_string())
                                    } else {
                                        Some(format!("row_{}", i))
                                    }
                                }
                                DataType::Utf8 => {
                                    if let Some(arr) = array
                                        .as_any()
                                        .downcast_ref::<datafusion::arrow::array::StringArray>(
                                    ) {
                                        Some(arr.value(i).to_string())
                                    } else {
                                        Some(format!("row_{}", i))
                                    }
                                }
                                _ => Some(format!("row_{}", i)),
                            };
                            string_values.push(value_str);
                        }
                    }
                    let string_array = StringArray::from(string_values);
                    Ok(ColumnarValue::Array(Arc::new(string_array)))
                }
            }
        }
    }
}

fn struct_to_csv_scalar(
    struct_array: &StructArray,
    options: &HashMap<String, String>,
) -> Result<ColumnarValue> {
    let mut field_values = Vec::new();
    for i in 0..struct_array.columns().len() {
        let column = struct_array.column(i);
        if column.len() > 0 && !column.is_null(0) {
            match column.data_type() {
                DataType::Boolean => {
                    if let Some(array) = column
                        .as_any()
                        .downcast_ref::<datafusion::arrow::array::BooleanArray>()
                    {
                        field_values.push(array.value(0).to_string());
                    }
                }
                DataType::Int32 => {
                    if let Some(array) = column
                        .as_any()
                        .downcast_ref::<datafusion::arrow::array::Int32Array>()
                    {
                        field_values.push(array.value(0).to_string());
                    }
                }
                DataType::Int64 => {
                    if let Some(array) = column
                        .as_any()
                        .downcast_ref::<datafusion::arrow::array::Int64Array>()
                    {
                        field_values.push(array.value(0).to_string());
                    }
                }
                DataType::Float64 => {
                    if let Some(array) = column
                        .as_any()
                        .downcast_ref::<datafusion::arrow::array::Float64Array>()
                    {
                        field_values.push(array.value(0).to_string());
                    }
                }
                DataType::Utf8 => {
                    if let Some(array) = column
                        .as_any()
                        .downcast_ref::<datafusion::arrow::array::StringArray>()
                    {
                        field_values.push(array.value(0).to_string());
                    }
                }
                DataType::Date32 => {
                    if let Some(array) = column
                        .as_any()
                        .downcast_ref::<datafusion::arrow::array::Date32Array>()
                    {
                        let days = array.value(0);
                        if let Ok(date_str) = conversion::format_date(days) {
                            field_values.push(date_str);
                        } else {
                            field_values.push(String::new());
                        }
                    }
                }
                DataType::Timestamp(_, _) => {
                    if let Some(array) = column
                        .as_any()
                        .downcast_ref::<datafusion::arrow::array::TimestampMicrosecondArray>(
                    ) {
                        let micros = array.value(0);
                        if let Ok(ts_str) = conversion::format_timestamp(micros) {
                            field_values.push(ts_str);
                        } else {
                            field_values.push(String::new());
                        }
                    }
                }
                _ => {
                    field_values.push(format!("{:?}", column.data_type()));
                }
            }
        } else {
            field_values.push(String::new());
        }
    }
    let delimiter = options
        .get("delimiter")
        .map(|s| s.chars().next().unwrap_or(','))
        .unwrap_or(',');

    let quote = options
        .get("quote")
        .map(|s| s.chars().next().unwrap_or('"'))
        .unwrap_or('"');

    let escape = options
        .get("escape")
        .map(|s| s.chars().next().unwrap_or('\\'))
        .unwrap_or('\\');
    let quoted_values: Vec<String> = field_values
        .iter()
        .map(|value| {
            if parsing::should_quote_csv_field(value, delimiter) {
                let escaped_value =
                    value.replace(quote.to_string().as_str(), &format!("{}{}", escape, quote));
                format!("{}{}{}", quote, escaped_value, quote)
            } else {
                value.clone()
            }
        })
        .collect();
    let csv_str = quoted_values.join(&delimiter.to_string());
    Ok(ColumnarValue::Scalar(ScalarValue::Utf8(Some(csv_str))))
}

fn process_struct_array_to_csv(
    struct_array: &StructArray,
    options: &HashMap<String, String>,
) -> Result<ColumnarValue> {
    let num_rows = struct_array.len();
    let mut csv_strings = Vec::with_capacity(num_rows);
    let delimiter = options
        .get("delimiter")
        .map(|s| s.chars().next().unwrap_or(','))
        .unwrap_or(',');
    let quote = options
        .get("quote")
        .map(|s| s.chars().next().unwrap_or('"'))
        .unwrap_or('"');
    let escape = options
        .get("escape")
        .map(|s| s.chars().next().unwrap_or('\\'))
        .unwrap_or('\\');
    for row_index in 0..num_rows {
        if struct_array.is_null(row_index) {
            csv_strings.push(None);
            continue;
        }
        let mut field_values = Vec::new();
        for i in 0..struct_array.columns().len() {
            let column = struct_array.column(i);
            if row_index < column.len() && !column.is_null(row_index) {
                match column.data_type() {
                    DataType::Boolean => {
                        if let Some(array) = column
                            .as_any()
                            .downcast_ref::<datafusion::arrow::array::BooleanArray>()
                        {
                            field_values.push(array.value(row_index).to_string());
                        }
                    }
                    DataType::Int32 => {
                        if let Some(array) = column
                            .as_any()
                            .downcast_ref::<datafusion::arrow::array::Int32Array>()
                        {
                            field_values.push(array.value(row_index).to_string());
                        }
                    }
                    DataType::Int64 => {
                        if let Some(array) = column
                            .as_any()
                            .downcast_ref::<datafusion::arrow::array::Int64Array>()
                        {
                            field_values.push(array.value(row_index).to_string());
                        }
                    }
                    DataType::Float64 => {
                        if let Some(array) = column
                            .as_any()
                            .downcast_ref::<datafusion::arrow::array::Float64Array>()
                        {
                            field_values.push(array.value(row_index).to_string());
                        }
                    }
                    DataType::Utf8 => {
                        if let Some(array) = column
                            .as_any()
                            .downcast_ref::<datafusion::arrow::array::StringArray>()
                        {
                            field_values.push(array.value(row_index).to_string());
                        }
                    }
                    DataType::Date32 => {
                        if let Some(array) = column
                            .as_any()
                            .downcast_ref::<datafusion::arrow::array::Date32Array>()
                        {
                            let days = array.value(row_index);
                            if let Ok(date_str) = conversion::format_date(days) {
                                field_values.push(date_str);
                            } else {
                                field_values.push(String::new());
                            }
                        }
                    }
                    DataType::Timestamp(_, _) => {
                        if let Some(array) = column
                            .as_any()
                            .downcast_ref::<datafusion::arrow::array::TimestampMicrosecondArray>(
                        ) {
                            let micros = array.value(row_index);
                            if let Ok(ts_str) = conversion::format_timestamp(micros) {
                                field_values.push(ts_str);
                            } else {
                                field_values.push(String::new());
                            }
                        }
                    }
                    _ => {
                        field_values.push(format!("{:?}", column.data_type()));
                    }
                }
            } else {
                field_values.push(String::new());
            }
        }
        let quoted_values: Vec<String> = field_values
            .iter()
            .map(|value| {
                if parsing::should_quote_csv_field(value, delimiter) {
                    let escaped_value =
                        value.replace(quote.to_string().as_str(), &format!("{}{}", escape, quote));
                    format!("{}{}{}", quote, escaped_value, quote)
                } else {
                    value.clone()
                }
            })
            .collect();
        let csv_str = quoted_values.join(&delimiter.to_string());
        csv_strings.push(Some(csv_str));
    }
    let string_array = StringArray::from(csv_strings);
    Ok(ColumnarValue::Array(Arc::new(string_array)))
}
