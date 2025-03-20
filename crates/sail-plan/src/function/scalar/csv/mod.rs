use datafusion::logical_expr::registry::FunctionRegistry;
use datafusion_expr;

mod conversion;
mod from_csv;
mod options;
mod parsing;
mod schema;
mod schema_of_csv;
mod to_csv;

use from_csv::from_csv;
use schema_of_csv::schema_of_csv;
use to_csv::to_csv;
use crate::function::common::ScalarFunction;

use std::sync::Arc;

// pub fn register_csv_functions(catalog: &mut dyn FunctionRegistry) -> Result<(), Box<dyn std::error::Error>> {
//     catalog.register_function(schema_of_csv::schema_of_csv)?;
//     catalog.register_function(from_csv::from_csv)?;
//     catalog.register_function(to_csv::to_csv)?;

//     catalog.register_udf(Arc::new(schema_of_csv::SchemaOfCsvUDF))?;
//     catalog.register_udf(Arc::new(from_csv::FromCsvUDF))?;
//     catalog.register_udf(Arc::new(to_csv::ToCsvUDF))?;

//     Ok(())
// }

pub(super) fn list_built_in_lambda_functions() -> Vec<(&'static str, ScalarFunction)> {
    use crate::function::common::ScalarFunctionBuilder as F;

    vec![
        ("schema_of_csv", F::custom(schema_of_csv)),
        ("from_csv", F::custom(from_csv)),
        ("to_csv", F::custom(to_csv)),
    ]
}
