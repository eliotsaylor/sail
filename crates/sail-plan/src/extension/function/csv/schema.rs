use std::sync::Arc;

use datafusion::arrow::array::{
    Array, BooleanBuilder, Float64Builder, Int32Builder, Int64Builder, StringBuilder, StructArray,
};
use datafusion::arrow::datatypes::{DataType, Field as ArrowField, Fields, TimeUnit};
use datafusion::common::{Result, ScalarValue};

use super::conversion;
use crate::error::{PlanError, PlanResult};

pub fn parse_schema(schema_str: &str) -> PlanResult<Vec<(String, String)>> {
    let schema_str = schema_str.trim();
    if !schema_str.to_uppercase().starts_with("STRUCT<") || !schema_str.ends_with(">") {
        return Err(PlanError::invalid(format!(
            "Invalid schema format: {}. Expected STRUCT<field1: TYPE1, field2: TYPE2, ...>",
            schema_str
        )));
    }
    let fields_part = &schema_str[schema_str.find('<').unwrap() + 1..schema_str.len() - 1];
    let mut fields = Vec::new();
    let mut start = 0;
    let mut depth = 0;
    let mut in_quotes = false;
    for (i, c) in fields_part.char_indices() {
        if c == '"' && fields_part.chars().nth(i.saturating_sub(1)) != Some('\\') {
            in_quotes = !in_quotes;
        } else if !in_quotes {
            if c == '<' {
                depth += 1;
            } else if c == '>' {
                depth -= 1;
            } else if c == ',' && depth == 0 {
                fields.push(fields_part[start..i].trim());
                start = i + 1;
            }
        }
    }
    if start < fields_part.len() {
        fields.push(fields_part[start..].trim());
    }
    let mut result = Vec::new();
    for field in fields {
        if let Some(colon_pos) = field.find(':') {
            let name = field[..colon_pos].trim().to_string();
            let type_str = field[colon_pos + 1..].trim().to_string();
            result.push((name, type_str));
        } else {
            return Err(PlanError::invalid(format!(
                "Invalid field format in schema: {}. Expected 'field: TYPE'",
                field
            )));
        }
    }
    Ok(result)
}

pub fn infer_field_types(fields: &[String]) -> Vec<String> {
    fields
        .iter()
        .map(|field| {
            // Try to infer types in this priority order: INT, DECIMAL, BOOLEAN, STRING
            if let Ok(_) = field.parse::<i32>() {
                "INT".to_string()
            } else if let Ok(_) = field.parse::<f64>() {
                "DECIMAL".to_string()
            } else if field.eq_ignore_ascii_case("true") || field.eq_ignore_ascii_case("false") {
                "BOOLEAN".to_string()
            } else {
                "STRING".to_string()
            }
        })
        .collect()
}

pub fn process_values_for_schema(
    values: &mut Vec<String>,
    struct_fields: &[(String, String)],
) -> Result<()> {
    if values.len() < struct_fields.len() {
        values.resize(struct_fields.len(), String::new());
    }
    for (i, (_, field_type)) in struct_fields.iter().enumerate() {
        if i < values.len() {
            let value = &values[i];
            if (value.starts_with('{') || value.starts_with('['))
                && field_type.to_uppercase() != "STRING"
            {
                match field_type.to_uppercase().as_str() {
                    "INT" => {
                        if i > 0 && values[i - 1].starts_with('{') && value.parse::<i32>().is_ok() {
                        } else {
                            values[i] = String::new();
                        }
                    }
                    "DATE" => {
                        if value.parse::<i32>().is_ok() {
                            if let Ok(year) = value.parse::<i32>() {
                                if year >= 1000 && year <= 9999 {
                                    values[i] = format!("{}-01-01", year);
                                } else {
                                    values[i] = String::new();
                                }
                            } else {
                                values[i] = String::new();
                            }
                        }
                    }
                    _ => {
                        values[i] = String::new();
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn create_struct_array(
    struct_fields: &[(String, String)],
    scalar_values: &[ScalarValue]
) -> Result<StructArray> {
    use datafusion::arrow::array::{ArrayRef, StructArray};
    use datafusion::arrow::datatypes::Field;
    use std::sync::Arc;
    let mut field_array_pairs = Vec::with_capacity(struct_fields.len());
    for ((name, type_str), scalar) in struct_fields.iter().zip(scalar_values.iter()) {
        let data_type = conversion::to_arrow_data_type(conversion::TypeInput::StringType(type_str));
        let field = Arc::new(Field::new(name, data_type.clone(), true));
        let array: ArrayRef = scalar.to_array_of_size(1)?;
        field_array_pairs.push((field, array));
    }
    let struct_array = StructArray::from(field_array_pairs);
    Ok(struct_array)
}

pub fn create_arrays_from_field_scalars(
    struct_fields: &[(String, &str)],
    field_arrays: &[Vec<Option<ScalarValue>>],
    _len: usize,
) -> Result<Vec<Arc<dyn Array>>> {
    let mut result = Vec::with_capacity(struct_fields.len());
    for (i, (_, field_type)) in struct_fields.iter().enumerate() {
        if i >= field_arrays.len() {
            return Err(datafusion::common::DataFusionError::Execution(format!(
                "Missing field array for field {}",
                i
            )));
        }
        let field_values = &field_arrays[i];
        let array = create_array_from_scalar_values(field_values, field_type)?;
        result.push(array);
    }

    Ok(result)
}

fn create_array_from_scalar_values(
    values: &[Option<ScalarValue>],
    type_str: &str,
) -> Result<Arc<dyn Array>> {
    match type_str.trim().to_uppercase().as_str() {
        "INT" | "INTEGER" => {
            let mut builder = Int32Builder::new();
            for value in values {
                match value {
                    Some(ScalarValue::Int32(Some(v))) => builder.append_value(*v),
                    Some(ScalarValue::Int32(None)) => builder.append_null(),
                    None => builder.append_null(),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as Arc<dyn Array>)
        }
        "BIGINT" | "LONG" => {
            let mut builder = Int64Builder::new();
            for value in values {
                match value {
                    Some(ScalarValue::Int64(Some(v))) => builder.append_value(*v),
                    Some(ScalarValue::Int64(None)) => builder.append_null(),
                    None => builder.append_null(),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as Arc<dyn Array>)
        }
        "DOUBLE" | "FLOAT" => {
            let mut builder = Float64Builder::new();
            for value in values {
                match value {
                    Some(ScalarValue::Float64(Some(v))) => builder.append_value(*v),
                    Some(ScalarValue::Float64(None)) => builder.append_null(),
                    None => builder.append_null(),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as Arc<dyn Array>)
        }
        "STRING" | "VARCHAR" | "CHAR" => {
            let mut builder = StringBuilder::new();
            for value in values {
                match value {
                    Some(ScalarValue::Utf8(Some(v))) => builder.append_value(v),
                    Some(ScalarValue::Utf8(None)) => builder.append_null(),
                    None => builder.append_null(),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as Arc<dyn Array>)
        }
        "BOOLEAN" | "BOOL" => {
            let mut builder = BooleanBuilder::new();
            for value in values {
                match value {
                    Some(ScalarValue::Boolean(Some(v))) => builder.append_value(*v),
                    Some(ScalarValue::Boolean(None)) => builder.append_null(),
                    None => builder.append_null(),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as Arc<dyn Array>)
        }
        _ => {
            let mut builder = StringBuilder::new();
            for value in values {
                match value {
                    Some(ScalarValue::Utf8(Some(v))) => builder.append_value(v),
                    Some(ScalarValue::Utf8(None)) => builder.append_null(),
                    None => builder.append_null(),
                    _ => match value {
                        Some(v) => builder.append_value(&v.to_string()),
                        None => builder.append_null(),
                    },
                }
            }
            Ok(Arc::new(builder.finish()) as Arc<dyn Array>)
        }
    }
}

pub fn create_struct_array_from_fields(
    struct_fields: &[(String, &str)],
    field_arrays: &[Arc<dyn Array>],
) -> Result<StructArray> {
    if struct_fields.len() != field_arrays.len() {
        return Err(datafusion::common::DataFusionError::Execution(format!(
            "Mismatched field count: {} fields, {} arrays",
            struct_fields.len(),
            field_arrays.len()
        )));
    }
    let fields: Vec<(Arc<ArrowField>, Arc<dyn Array>)> = struct_fields
        .iter()
        .enumerate()
        .map(|(i, (name, type_str))| {
            let field = Arc::new(ArrowField::new(
                name,
                match type_str.trim().to_uppercase().as_str() {
                    "INT" | "INTEGER" => DataType::Int32,
                    "BIGINT" | "LONG" => DataType::Int64,
                    "DOUBLE" | "FLOAT" => DataType::Float64,
                    "STRING" | "VARCHAR" | "CHAR" => DataType::Utf8,
                    "BOOLEAN" | "BOOL" => DataType::Boolean,
                    _ => DataType::Utf8,
                },
                true,
            ));
            (field, field_arrays[i].clone())
        })
        .collect();
    Ok(StructArray::from(fields))
}

pub fn convert_schema_to_arrow_fields(
    struct_fields: &[(String, String)]
) -> Result<Fields> {
    let mut fields = Vec::with_capacity(struct_fields.len());

    for (name, type_str) in struct_fields {
        let data_type = match type_str.trim().to_uppercase().as_str() {
            "INT" | "INTEGER" => DataType::Int32,
            "BIGINT" | "LONG" => DataType::Int64,
            "DOUBLE" | "FLOAT" => DataType::Float64,
            "BOOLEAN" | "BOOL" => DataType::Boolean,
            "STRING" | "VARCHAR" | "CHAR" => DataType::Utf8,
            "DATE" => DataType::Date32,
            "TIMESTAMP" => DataType::Timestamp(TimeUnit::Microsecond, None),
            _ => DataType::Utf8,
        };

        fields.push(ArrowField::new(name, data_type, true));
    }

    Ok(Fields::from(fields))
}
