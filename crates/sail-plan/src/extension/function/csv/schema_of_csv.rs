use std::any::Any;
use std::collections::HashMap;
use std::sync::LazyLock;

use datafusion::arrow::array::Array;
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::{lit, Expr};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};

use super::{options, parsing, schema};
use crate::error::{PlanError, PlanResult};
use crate::function::common::ScalarFunctionInput;
use crate::utils::ItemTaker;

#[derive(Debug)]
pub struct SchemaOfCsvUDF;

impl ScalarUDFImpl for SchemaOfCsvUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "schema_of_csv"
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
                "schema_of_csv requires at least 1 argument: csv_string".to_string(),
            ));
        }
        let csv_input = &args.args[0];
        let options_map = if args.args.len() > 1 {
            options::extract_options_from_columnar(&args.args[1])?
        } else {
            HashMap::new()
        };
        match csv_input {
            ColumnarValue::Scalar(ScalarValue::Utf8(Some(csv_str))) => {
                let fields = parsing::parse_csv_line_df(csv_str, &options_map)?;
                let field_types = schema::infer_field_types(&fields);
                let schema_parts: Vec<String> = fields
                    .iter()
                    .enumerate()
                    .zip(field_types.iter())
                    .map(|((i, _), field_type)| format!("_c{}: {}", i, field_type))
                    .collect();
                let schema_ddl = format!("STRUCT<{}>", schema_parts.join(", "));
                Ok(ColumnarValue::Scalar(ScalarValue::Utf8(Some(schema_ddl))))
            }
            ColumnarValue::Array(array) => {
                use datafusion::arrow::array::StringArray;
                if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
                    if string_array.len() > 0 && !string_array.is_null(0) {
                        let csv_str = string_array.value(0);
                        let fields = parsing::parse_csv_line_df(csv_str, &options_map)?;
                        let field_types = schema::infer_field_types(&fields);
                        let schema_parts: Vec<String> = fields
                            .iter()
                            .enumerate()
                            .zip(field_types.iter())
                            .map(|((i, _), field_type)| format!("_c{}: {}", i, field_type))
                            .collect();
                        let schema_ddl = format!("STRUCT<{}>", schema_parts.join(", "));
                        Ok(ColumnarValue::Scalar(ScalarValue::Utf8(Some(schema_ddl))))
                    } else {
                        Ok(ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                            "STRUCT<>".to_string(),
                        ))))
                    }
                } else {
                    Err(datafusion::common::DataFusionError::Execution(
                        "Input to schema_of_csv must be a string array".to_string(),
                    ))
                }
            }
            _ => Err(datafusion::common::DataFusionError::Execution(
                "Input to schema_of_csv must be a string".to_string(),
            )),
        }
    }
}
