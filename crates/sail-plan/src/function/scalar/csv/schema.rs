use std::sync::Arc;

use datafusion::arrow::array::{
    Array, BooleanBuilder, Float64Builder, Int32Builder, Int64Builder, StringBuilder, StructArray,
};
use datafusion::arrow::datatypes::{
    DataType as ArrowDataType, DataType, Field as ArrowField, Field,
};
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
    values: &[ScalarValue],
) -> Result<StructArray> {
    let mut field_arrays = Vec::with_capacity(struct_fields.len());
    for (i, (field_name, field_type)) in struct_fields.iter().enumerate() {
        let scalar_value = if i < values.len() {
            values[i].clone()
        } else {
            conversion::create_null_scalar_value(field_type)?
        };
        let arrow_type =
            conversion::to_arrow_data_type(conversion::TypeInput::StringType(field_type));
        let field = Arc::new(ArrowField::new(field_name, arrow_type, true));
        let array = scalar_value.to_array()?;
        field_arrays.push((field, array));
    }

    Ok(StructArray::from(
        field_arrays
            .iter()
            .map(|(field, array)| (Arc::clone(field), Arc::clone(array)))
            .collect::<Vec<_>>(),
    ))
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
    let fields: Vec<(Arc<Field>, Arc<dyn Array>)> = struct_fields
        .iter()
        .enumerate()
        .map(|(i, (name, type_str))| {
            let field = Arc::new(Field::new(
                name,
                match type_str.trim().to_uppercase().as_str() {
                    "INT" | "INTEGER" => DataType::Int32,
                    "BIGINT" | "LONG" => DataType::Int64,
                    "DOUBLE" | "FLOAT" => DataType::Float64,
                    "STRING" | "VARCHAR" | "CHAR" => DataType::Utf8,
                    "BOOLEAN" | "BOOL" => DataType::Boolean,
                    _ => DataType::Utf8,
                },
                true, // nullable
            ));
            (field, field_arrays[i].clone())
        })
        .collect();
    Ok(StructArray::from(fields))
}
