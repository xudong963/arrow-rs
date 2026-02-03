// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Tests for skipping predicate evaluation on fully-matched row groups.
//!
//! When statistics prove that ALL rows in a row group match the filter,
//! predicate evaluation can be skipped entirely for that row group,
//! saving CPU by avoiding filter column decoding and predicate evaluation.
//!
//! Note: This optimization only applies to the async (push-decoder) path,
//! which processes row groups individually. The sync reader evaluates
//! predicates across all row groups in a single pass.

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use parquet::arrow::arrow_reader::{
    ArrowPredicateFn, ArrowReaderOptions, RowFilter,
};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::{ArrowWriter, ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use parquet::file::properties::WriterProperties;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

/// Create a parquet file in memory with 3 row groups of 200 rows each.
/// Column "a" contains i64 values 0..600.
static TEST_FILE_DATA: LazyLock<Bytes> = LazyLock::new(|| {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

    let mut output = Vec::new();
    let props = WriterProperties::builder()
        .set_max_row_group_size(200)
        .build();
    let mut writer =
        ArrowWriter::try_new(&mut output, Arc::clone(&schema), Some(props)).unwrap();

    // Write 600 rows total -> 3 row groups of 200 each
    let values: Vec<i64> = (0..600).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(values)) as ArrayRef],
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    Bytes::from(output)
});

/// Builds a counting predicate that filters `a > 100` and increments a counter
/// each time `evaluate()` is called.
fn make_counting_filter(
    call_count: Arc<AtomicUsize>,
    mask: ProjectionMask,
) -> ArrowPredicateFn<impl FnMut(RecordBatch) -> Result<BooleanArray, arrow_schema::ArrowError>> {
    ArrowPredicateFn::new(mask, move |batch: RecordBatch| {
        call_count.fetch_add(1, Ordering::SeqCst);
        let col = batch.column(0).as_primitive::<Int64Type>();
        let scalar = Int64Array::new_scalar(100);
        arrow::compute::kernels::cmp::gt(col, &scalar)
    })
}

/// Async reader for tests (same pattern as predicate_cache.rs)
#[derive(Clone)]
struct TestReader {
    data: Bytes,
    metadata: Option<Arc<ParquetMetaData>>,
}

impl TestReader {
    fn new(data: Bytes) -> Self {
        Self {
            data,
            metadata: None,
        }
    }
}

impl AsyncFileReader for TestReader {
    fn get_bytes(
        &mut self,
        range: std::ops::Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let data = self.data.slice(range.start as usize..range.end as usize);
        futures::future::ready(Ok(data)).boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        let mut reader = ParquetMetaDataReader::new();
        if let Some(options) = options {
            reader = reader
                .with_column_index_policy(options.column_index_policy())
                .with_offset_index_policy(options.offset_index_policy());
        }
        self.metadata = Some(Arc::new(reader.parse_and_finish(&self.data).unwrap()));
        futures::future::ready(Ok(self.metadata.clone().unwrap())).boxed()
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

/// Test that predicate evaluation is skipped for row groups marked as
/// fully matched (skip RG 0 and RG 2), while still applied to RG 1.
#[tokio::test]
async fn test_skip_predicate_for_fully_matched_row_groups() {
    let data = TEST_FILE_DATA.clone();
    let call_count = Arc::new(AtomicUsize::new(0));

    let reader = TestReader::new(data);
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        reader,
        ArrowReaderOptions::default(),
    )
    .await
    .unwrap();

    // Verify we have 3 row groups
    assert_eq!(builder.metadata().num_row_groups(), 3);

    let schema_descr = builder.metadata().file_metadata().schema_descr_ptr();
    let mask = ProjectionMask::leaves(&schema_descr, vec![0]);
    let filter = make_counting_filter(Arc::clone(&call_count), mask);
    let row_filter =
        RowFilter::new(vec![Box::new(filter)]).with_skip_row_groups(HashSet::from([0, 2]));

    let mut stream = builder.with_row_filter(row_filter).build().unwrap();

    let mut total_rows = 0;
    while let Some(batch) = stream.next().await {
        total_rows += batch.unwrap().num_rows();
    }

    // RG 0 (values 0..200): skipped -> all 200 rows returned (no filtering)
    // RG 1 (values 200..400): filtered by a > 100 -> all 200 match -> 200 rows
    // RG 2 (values 400..600): skipped -> all 200 rows returned (no filtering)
    assert_eq!(total_rows, 600, "All rows should be returned");

    // The predicate should only be called for RG 1, not for RG 0 or RG 2.
    let calls = call_count.load(Ordering::SeqCst);
    assert_eq!(
        calls, 1,
        "Predicate should be called exactly once (only for RG 1)"
    );
}

/// Test that when ALL row groups are skipped, the predicate is never called
/// and all rows are returned.
#[tokio::test]
async fn test_all_row_groups_skipped() {
    let data = TEST_FILE_DATA.clone();
    let call_count = Arc::new(AtomicUsize::new(0));

    let reader = TestReader::new(data);
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        reader,
        ArrowReaderOptions::default(),
    )
    .await
    .unwrap();

    let schema_descr = builder.metadata().file_metadata().schema_descr_ptr();
    let mask = ProjectionMask::leaves(&schema_descr, vec![0]);
    let filter = make_counting_filter(Arc::clone(&call_count), mask);
    let row_filter = RowFilter::new(vec![Box::new(filter)])
        .with_skip_row_groups(HashSet::from([0, 1, 2]));

    let mut stream = builder.with_row_filter(row_filter).build().unwrap();

    let mut total_rows = 0;
    while let Some(batch) = stream.next().await {
        total_rows += batch.unwrap().num_rows();
    }

    assert_eq!(total_rows, 600, "All rows should be returned");
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        0,
        "Predicate should never be called when all row groups are skipped"
    );
}

/// Test that with no skip set (default), all row groups are filtered normally.
#[tokio::test]
async fn test_no_row_groups_skipped() {
    let data = TEST_FILE_DATA.clone();
    let call_count = Arc::new(AtomicUsize::new(0));

    let reader = TestReader::new(data);
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        reader,
        ArrowReaderOptions::default(),
    )
    .await
    .unwrap();

    let schema_descr = builder.metadata().file_metadata().schema_descr_ptr();
    let mask = ProjectionMask::leaves(&schema_descr, vec![0]);
    let filter = make_counting_filter(Arc::clone(&call_count), mask);
    // No skip_row_groups -> empty skip set
    let row_filter = RowFilter::new(vec![Box::new(filter)]);

    let mut stream = builder.with_row_filter(row_filter).build().unwrap();

    let mut total_rows = 0;
    while let Some(batch) = stream.next().await {
        total_rows += batch.unwrap().num_rows();
    }

    // a > 100: RG 0 has 0..200 -> values 101..199 = 99 rows match
    // RG 1 has 200..400 -> all 200 match
    // RG 2 has 400..600 -> all 200 match
    assert_eq!(
        total_rows, 499,
        "Only rows with a > 100 should be returned"
    );

    // All 3 row groups should have their predicates evaluated
    let calls = call_count.load(Ordering::SeqCst);
    assert_eq!(
        calls, 3,
        "Predicate should be called for all 3 row groups"
    );
}

/// Test that with_skip_row_groups(empty set) is equivalent to no skip.
#[tokio::test]
async fn test_empty_skip_set() {
    let data = TEST_FILE_DATA.clone();
    let call_count = Arc::new(AtomicUsize::new(0));

    let reader = TestReader::new(data);
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        reader,
        ArrowReaderOptions::default(),
    )
    .await
    .unwrap();

    let schema_descr = builder.metadata().file_metadata().schema_descr_ptr();
    let mask = ProjectionMask::leaves(&schema_descr, vec![0]);
    let filter = make_counting_filter(Arc::clone(&call_count), mask);
    let row_filter =
        RowFilter::new(vec![Box::new(filter)]).with_skip_row_groups(HashSet::new());

    let mut stream = builder.with_row_filter(row_filter).build().unwrap();

    let mut total_rows = 0;
    while let Some(batch) = stream.next().await {
        total_rows += batch.unwrap().num_rows();
    }

    assert_eq!(
        total_rows, 499,
        "Only rows with a > 100 should be returned"
    );
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        3,
        "Predicate should be called for all 3 row groups with empty skip set"
    );
}
