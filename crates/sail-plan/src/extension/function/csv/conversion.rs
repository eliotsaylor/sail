use chrono::{Datelike, NaiveDate, TimeZone};
use datafusion::arrow::datatypes::{
    DataType as ArrowDataType, Field as ArrowField, TimeUnit as ArrowTimeUnit,
};
use datafusion::common::{DataFusionError, Result, ScalarValue};
use sail_common::spec::DataType;

use crate::error::PlanResult;

pub enum TypeInput<'a> {
    StringType(&'a str),
    #[allow(dead_code)]
    SailType(&'a DataType),
    #[allow(dead_code)]
    ScalarValue(&'a ScalarValue),
}

pub fn to_arrow_data_type(input: TypeInput) -> ArrowDataType {
    match input {
        TypeInput::StringType(type_str) => match type_str.to_uppercase().as_str() {
            "INT" => ArrowDataType::Int32,
            "DOUBLE" => ArrowDataType::Float64,
            "BOOLEAN" => ArrowDataType::Boolean,
            "STRING" => ArrowDataType::Utf8,
            "DATE" => ArrowDataType::Date32,
            "TIMESTAMP" => ArrowDataType::Timestamp(ArrowTimeUnit::Microsecond, None),
            _ => ArrowDataType::Utf8,
        },
        TypeInput::SailType(sail_type) => match sail_type {
            DataType::Int32 => ArrowDataType::Int32,
            DataType::Int64 => ArrowDataType::Int64,
            DataType::Float64 => ArrowDataType::Float64,
            DataType::Boolean => ArrowDataType::Boolean,
            DataType::Utf8 => ArrowDataType::Utf8,
            DataType::Date32 => ArrowDataType::Date32,
            DataType::Timestamp {
                time_unit: _,
                timezone_info: _,
            } => ArrowDataType::Timestamp(ArrowTimeUnit::Microsecond, None),
            _ => ArrowDataType::Utf8,
        },
        TypeInput::ScalarValue(value) => match value {
            ScalarValue::Boolean(_) => ArrowDataType::Boolean,
            ScalarValue::Int32(_) => ArrowDataType::Int32,
            ScalarValue::Int64(_) => ArrowDataType::Int64,
            ScalarValue::Float64(_) => ArrowDataType::Float64,
            ScalarValue::Utf8(_) => ArrowDataType::Utf8,
            ScalarValue::Date32(_) => ArrowDataType::Date32,
            ScalarValue::TimestampMicrosecond(_, tz) => {
                ArrowDataType::Timestamp(ArrowTimeUnit::Microsecond, tz.clone())
            }
            _ => ArrowDataType::Null,
        },
    }
}

#[allow(dead_code)]
pub fn create_arrow_fields(struct_fields: &[(String, String)]) -> PlanResult<Vec<ArrowField>> {
    let mut fields = Vec::with_capacity(struct_fields.len());
    for (field_name, field_type) in struct_fields {
        let arrow_type = to_arrow_data_type(TypeInput::StringType(field_type));
        fields.push(ArrowField::new(field_name, arrow_type, true));
    }
    Ok(fields)
}

pub fn create_null_scalar_value(data_type: &str) -> Result<ScalarValue> {
    match data_type.to_uppercase().as_str() {
        "INT" => Ok(ScalarValue::Int32(None)),
        "DOUBLE" => Ok(ScalarValue::Float64(None)),
        "BOOLEAN" => Ok(ScalarValue::Boolean(None)),
        "STRING" => Ok(ScalarValue::Utf8(None)),
        "DATE" => Ok(ScalarValue::Date32(None)),
        "TIMESTAMP" => Ok(ScalarValue::TimestampMicrosecond(None, None)),
        _ => Err(DataFusionError::Execution(format!(
            "Unsupported data type: {}",
            data_type
        ))),
    }
}

pub fn convert_csv_value(value: &str, data_type: &str) -> Result<ScalarValue> {
    let trimmed = value.trim();
    match data_type.to_uppercase().as_str() {
        "INT" => match trimmed.parse::<i32>() {
            Ok(num) => Ok(ScalarValue::Int32(Some(num))),
            Err(e) => Err(DataFusionError::Execution(format!(
                "Failed to parse '{}' as INT: {}",
                value, e
            ))),
        },
        "DOUBLE" => match trimmed.parse::<f64>() {
            Ok(num) => Ok(ScalarValue::Float64(Some(num))),
            Err(e) => Err(DataFusionError::Execution(format!(
                "Failed to parse '{}' as DOUBLE: {}",
                value, e
            ))),
        },
        "BOOLEAN" => match trimmed.to_lowercase().as_str() {
            "true" | "t" | "yes" | "y" | "1" => Ok(ScalarValue::Boolean(Some(true))),
            "false" | "f" | "no" | "n" | "0" => Ok(ScalarValue::Boolean(Some(false))),
            _ => Err(DataFusionError::Execution(format!(
                "Failed to parse '{}' as BOOLEAN",
                value
            ))),
        },
        "STRING" => Ok(ScalarValue::Utf8(Some(trimmed.to_string()))),
        "DATE" => {
            if trimmed.len() == 10 && trimmed.matches('-').count() == 2 {
                let parts: Vec<&str> = trimmed.split('-').collect();
                if parts.len() == 3 {
                    if let (Ok(year), Ok(month), Ok(day)) = (
                        parts[0].parse::<i32>(),
                        parts[1].parse::<u32>(),
                        parts[2].parse::<u32>(),
                    ) {
                        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                            let days = date.ordinal() as i32 - epoch.ordinal() as i32;
                            return Ok(ScalarValue::Date32(Some(days)));
                        }
                    }
                }
            }
            Err(DataFusionError::Execution(format!(
                "Failed to parse '{}' as DATE",
                value
            )))
        }
        "TIMESTAMP" => {
            let datetime_formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y/%m/%d %H:%M:%S",
            ];
            for format in &datetime_formats {
                if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(trimmed, format) {
                    #[allow(deprecated)]
                    let micros = dt.timestamp_micros();
                    return Ok(ScalarValue::TimestampMicrosecond(Some(micros), None));
                }
            }
            let time_formats = ["%H:%M:%S", "%I:%M:%S %p"];
            for format in &time_formats {
                if let Ok(time) = chrono::NaiveTime::parse_from_str(trimmed, format) {
                    let today = chrono::Local::now().date_naive();
                    let dt = today.and_time(time);
                    #[allow(deprecated)]
                    let micros = dt.timestamp_micros();
                    return Ok(ScalarValue::TimestampMicrosecond(Some(micros), None));
                }
            }
            Err(DataFusionError::Execution(format!(
                "Failed to parse '{}' as TIMESTAMP",
                value
            )))
        }
        _ => Err(DataFusionError::Execution(format!(
            "Unsupported data type: {}",
            data_type
        ))),
    }
}

pub fn format_date(days_since_epoch: i32) -> Result<String> {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)
        .ok_or_else(|| DataFusionError::Internal("Invalid epoch date".to_string()))?;
    let date = epoch + chrono::Duration::days(days_since_epoch as i64);
    Ok(date.format("%Y-%m-%d").to_string())
}

pub fn format_timestamp(micros: i64) -> Result<String> {
    let secs = micros / 1_000_000;
    let nsecs = (micros % 1_000_000) * 1_000;
    let datetime = chrono::Utc
        .timestamp_opt(secs, nsecs as u32)
        .single()
        .ok_or_else(|| {
            DataFusionError::Internal(format!("Invalid timestamp value: {} microseconds", micros))
        })?;
    Ok(datetime.format("%Y-%m-%d %H:%M:%S").to_string())
}
