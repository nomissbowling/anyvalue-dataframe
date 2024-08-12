#![doc(html_root_url = "https://docs.rs/anyvalue-dataframe/0.1.0")]
//! anyvalue dataframe
//!

use std::error::Error;
use polars::prelude::{DataFrame, AnyValue, Schema, Field, DataType};

/// to anyvalue from value and datatype
/// let a = to_any!(3, DataType::UInt64);
/// let b = to_any!("X", DataType::Utf8);
#[macro_export]
macro_rules! to_any {
  ($v: expr, DataType::Null) => { AnyValue::Null };
  // Date: feature dtype-date
  // Time: feature dtype-date
  // DataType:: DateTime, Duration, Categorical, List, Object, Struct
  //   feature dtype-datetime -duration -categorical -array
  // AnyValue:: Enum, Array, Decimal, xxxOwned, etc
  ($v: expr, DataType:: $t: ident) => { AnyValue::$t($v) }
}
// pub to_any;

/// row schema from vec AnyValue (column names are column_0, column_1, ...)
/// - let schema = Schema::from(&row);
pub fn row_schema(row: Vec<AnyValue<'_>>) -> polars::frame::row::Row {
  polars::frame::row::Row::new(row)
}

/// row fields from vec (&amp;str, DataType) (set with column names)
/// - let schema = Schema::from_iter(fields);
pub fn row_fields(desc: Vec<(&str, DataType)>) -> Vec<Field> {
  desc.into_iter().map(|(s, t)| Field::new(s, t)).collect()
}

/// named fields from DataFrame
pub fn named_fields(df: &DataFrame, n: Vec<&str>) -> Vec<Field> {
  let t = df.dtypes();
  row_fields(n.into_iter().enumerate().map(|(i, s)|
    (s, t[i].clone())).collect())
}

/// named schema from DataFrame
/// - same as df.schema() after column names are set by df.set_column_names()
/// - otherwise df.schema() returns names as column_0, column_1, ...
pub fn named_schema(df: &DataFrame, n: Vec<&str>) -> Schema {
  Schema::from_iter(named_fields(&df, n))
}

/// DataFrame from Vec&lt;polars::frame::row::Row&gt; and field names
pub fn df_from_vec(rows: &Vec<polars::frame::row::Row>, n: &Vec<&str>) ->
  Result<DataFrame, Box<dyn Error>> {
  let schema = Schema::from(&rows[0]);
  let mut df = DataFrame::from_rows_iter_and_schema(rows.iter(), &schema)?;
  df.set_column_names(&n)?;
  Ok(df)
}

/// tests
#[cfg(test)]
mod tests {
  use super::*;

  /// [-- --nocapture] [-- --show-output]
  #[test]
  fn test_dataframe() {
    let a = to_any!(3, DataType::UInt64);
    assert_eq!(a, AnyValue::UInt64(3));
    assert_eq!(a.dtype(), DataType::UInt64);
    let b = to_any!("A", DataType::Utf8);
    assert_eq!(b, AnyValue::Utf8("A"));
    assert_eq!(b.dtype(), DataType::Utf8);
    let c = to_any!(4, DataType::Int8);
    assert_eq!(c, AnyValue::Int8(4));
    assert_eq!(c.dtype(), DataType::Int8);
    let d = to_any!(1.5, DataType::Float64);
    assert_eq!(d, AnyValue::Float64(1.5));
    assert_eq!(d.dtype(), DataType::Float64);
    let e = to_any!(true, DataType::Boolean);
    assert_eq!(e, AnyValue::Boolean(true));
    assert_eq!(e.dtype(), DataType::Boolean);
    let f = to_any!(&[255, 0], DataType::Binary);
    assert_eq!(f, AnyValue::Binary(&[255, 0]));
    assert_eq!(f.dtype(), DataType::Binary);
  }
}