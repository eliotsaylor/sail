
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::{lit, Expr};
use crate::error::{PlanError, PlanResult};
use crate::function::common::ScalarFunctionInput;


/// Infers the schema of a CSV string and returns it in DDL format.
///
/// Arguments:
///   - csv_str: A string literal containing CSV data. The function expects a
///     single line of CSV data.
///   - options: An optional map of CSV parsing options. Supported options include:
///     - delimiter: The character used to separate fields (default: ',')
///     - quote: The character used for quoting (default: '"')
///     - escape: The character used for escaping (default: '\')
///
/// Returns:
///   - A string literal in DDL format representing the inferred schema, in the form:
///     "STRUCT<_c0: TYPE1, _c1: TYPE2, ...>"
fn schema_of_csv(input: ScalarFunctionInput) -> PlanResult<Expr> {
    let ScalarFunctionInput { arguments, .. } = input;
    let (csv_expr, option_expr) = match arguments.len() {
        1 => (arguments.one()?, None),
        2 => (arguments[0].clone(), Some(arguments[1].clone())),
        _ => return Err(PlanError::todo("schema_of_csv expects 1 or 2 arguments")),
    };
    let csv_str = match csv_expr {
        Expr::Literal(ScalarValue::Utf8(Some(csv))) => csv,
        _ => {
            return Err(PlanError::todo(
                "schema_of_csv requires a foldable string input",
            ))
        }
    };
    let options_map = if let Some(opt_expr) = &option_expr {
        options::extract_options_from_expr(opt_expr)?
    } else {
        HashMap::new()
    };
    let fields = parsing::parse_csv_line_df(&csv_str, &options_map)?;
    let field_types = schema::infer_field_types(&fields);
    let schema_parts: Vec<String> = fields
        .iter()
        .enumerate()
        .zip(field_types.iter())
        .map(|((i, _), field_type)| format!("_c{}: {}", i, field_type))
        .collect();
    let schema_ddl = format!("STRUCT<{}>", schema_parts.join(", "));
    Ok(lit(ScalarValue::Utf8(Some(schema_ddl))))
}

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
fn from_csv(input: ScalarFunctionInput) -> PlanResult<Expr> {
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

/// Converts a column containing a struct type into a CSV string.
///
/// Arguments:
///   - struct_col: A column containing a struct type to be converted to CSV.
///   - options: An optional map of CSV formatting options. Supported options include:
///     - delimiter: The character used to separate fields (default: ',')
///     - quote: The character used for quoting (default: '"')
///     - escape: The character used for escaping (default: '\')
///     - dateFormat: The format string for date values (default: 'yyyy-MM-dd')
///     - timestampFormat: The format string for timestamp values (default: 'yyyy-MM-dd HH:mm:ss')
///
/// Returns:
///   - A string containing the CSV representation of the struct.
fn to_csv(input: ScalarFunctionInput) -> PlanResult<Expr> {
    let ScalarFunctionInput { arguments, .. } = input;
    let (struct_expr, options_expr) = match arguments.len() {
        1 => (arguments.one()?, None),
        2 => {
            let struct_expr = arguments[0].clone();
            let opts = Some(arguments[1].clone());
            (struct_expr, opts)
        }
        _ => return Err(PlanError::todo("to_csv expects 1 or 2 arguments")),
    };

    if let Expr::Literal(ScalarValue::Struct(struct_array)) = &struct_expr {
        let options_map = if let Some(opts) = &options_expr {
            options::extract_options_from_expr(opts)?
        } else {
            HashMap::new()
        };
        let field_values = extract_struct_field_values(struct_array, &options_map)?;
        let csv_str = format_as_csv(&field_values, &options_map)?;
        return Ok(lit(ScalarValue::Utf8(Some(csv_str))));
    }
    let options_expr = if let Some(opts) = options_expr {
        opts
    } else {
        Expr::Literal(ScalarValue::Utf8(Some(String::new())))
    };
    let udf = Arc::new(datafusion_expr::ScalarUDF::new_from_impl(ToCsvUDF));
    Ok(datafusion_expr::expr::Expr::ScalarFunction(
        datafusion_expr::expr::ScalarFunction::new_udf(udf, vec![struct_expr, options_expr]),
    ))
}

pub(super) fn list_built_in_hash_functions() -> Vec<(&'static str, ScalarFunction)> {
    use crate::function::common::ScalarFunctionBuilder as F;

    vec![
        ("schema_of_csv", F::custom(schema_of_csv)),
        ("from_csv", F::custom(from_csv)),
        ("to_csv", F::custom(to_csv)),
    ]
}
