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

#[allow(dead_code)]
pub fn register_csv_functions(registry: &mut impl FunctionRegistry) {
    let schema_of_csv_udf =
        datafusion_expr::ScalarUDF::new_from_impl(schema_of_csv::SchemaOfCsvUDF);
    let from_csv_udf = datafusion_expr::ScalarUDF::new_from_impl(from_csv::FromCsvUDF);
    let to_csv_udf = datafusion_expr::ScalarUDF::new_from_impl(to_csv::ToCsvUDF);
    registry
        .register_udf(std::sync::Arc::new(schema_of_csv_udf))
        .expect("Failed to register schema_of_csv");
    registry
        .register_udf(std::sync::Arc::new(from_csv_udf))
        .expect("Failed to register from_csv");
    registry
        .register_udf(std::sync::Arc::new(to_csv_udf))
        .expect("Failed to register to_csv");
}

pub(super) fn list_built_in_csv_functions(
) -> Vec<(&'static str, crate::function::common::ScalarFunction)> {
    use crate::function::common::ScalarFunctionBuilder as F;

    vec![
        ("schema_of_csv", F::custom(schema_of_csv)),
        ("from_csv", F::custom(from_csv)),
        ("to_csv", F::custom(to_csv)),
    ]
}
