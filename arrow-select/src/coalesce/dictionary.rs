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

use super::InProgressArray;
use crate::concat::concat;
use crate::filter::{FilterPredicate, FilterSelection};
use arrow_array::builder::StringDictionaryBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, DictionaryArray, StringArray};
use arrow_schema::ArrowError;
use std::sync::Arc;

pub(crate) struct InProgressStringDictionaryArray {
    source: Option<ArrayRef>,
    batch_size: usize,
    builder: Option<StringDictionaryBuilder<Int32Type>>,
    buffered_arrays: Vec<ArrayRef>,
    source_data_referenced_in_buffers: bool,
    total_size_of_non_shared_buffers: usize,
    builder_rows: usize,
    builder_value_bytes: usize,
}

impl std::fmt::Debug for InProgressStringDictionaryArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InProgressStringDictionaryArray")
            .field("batch_size", &self.batch_size)
            .field("builder_rows", &self.builder_rows)
            .field("buffered_arrays", &self.buffered_arrays.len())
            .finish()
    }
}

impl InProgressStringDictionaryArray {
    pub(crate) fn new(batch_size: usize) -> Self {
        Self {
            source: None,
            batch_size,
            builder: None,
            buffered_arrays: Vec::new(),
            source_data_referenced_in_buffers: false,
            total_size_of_non_shared_buffers: 0,
            builder_rows: 0,
            builder_value_bytes: 0,
        }
    }

    fn append_index(
        builder: &mut StringDictionaryBuilder<Int32Type>,
        source: &DictionaryArray<Int32Type>,
        values: &StringArray,
        index: usize,
    ) -> Result<usize, ArrowError> {
        match source.key(index) {
            Some(key) => {
                let value = values.value(key);
                builder.append(value)?;
                Ok(value.len())
            }
            None => {
                builder.append_null();
                Ok(0)
            }
        }
    }

    fn flush_builder(&mut self) {
        let Some(mut builder) = self.builder.take() else {
            return;
        };
        let array = Arc::new(builder.finish()) as ArrayRef;
        self.total_size_of_non_shared_buffers += array.get_array_memory_size();
        self.buffered_arrays.push(array);
        self.builder_rows = 0;
        self.builder_value_bytes = 0;
    }

    fn buffer_materialized_filter(
        &mut self,
        source: &dyn Array,
        filter: &FilterPredicate,
    ) -> Result<(), ArrowError> {
        self.flush_builder();
        let filtered = filter.filter(source)?;
        self.total_size_of_non_shared_buffers += filtered.get_array_memory_size();
        self.buffered_arrays.push(filtered);
        Ok(())
    }
}

impl InProgressArray for InProgressStringDictionaryArray {
    fn set_source(&mut self, source: Option<ArrayRef>) {
        if let Some(old_source) = self.source.take()
            && self.source_data_referenced_in_buffers
        {
            self.total_size_of_non_shared_buffers += old_source.get_array_memory_size();
        }
        self.source_data_referenced_in_buffers = false;
        self.source = source;
    }

    fn copy_rows(&mut self, offset: usize, len: usize) -> Result<(), ArrowError> {
        self.flush_builder();
        let source = self.source.as_ref().ok_or_else(|| {
            ArrowError::InvalidArgumentError(
                "Internal Error: InProgressStringDictionaryArray: source not set".to_string(),
            )
        })?;
        self.source_data_referenced_in_buffers = true;
        self.buffered_arrays.push(source.slice(offset, len));
        Ok(())
    }

    fn copy_rows_by_filter_from(
        &mut self,
        source: ArrayRef,
        filter: &FilterPredicate,
    ) -> Result<(), ArrowError> {
        let dictionary = source.as_dictionary::<Int32Type>();
        let values = dictionary.values().as_string::<i32>();

        if values.null_count() != 0 {
            return self.buffer_materialized_filter(source.as_ref(), filter);
        }

        if values.len() <= 256 && filter.count() > filter.filter_len() / 128 {
            return self.buffer_materialized_filter(source.as_ref(), filter);
        }

        let builder = self.builder.get_or_insert_with(|| {
            StringDictionaryBuilder::with_capacity(
                self.batch_size,
                values.len(),
                values.get_array_memory_size(),
            )
        });

        let mut appended_value_bytes = 0;
        match filter.selection() {
            FilterSelection::None => {}
            FilterSelection::All { len } => {
                for index in 0..len {
                    appended_value_bytes += Self::append_index(builder, dictionary, values, index)?;
                }
            }
            FilterSelection::Slices(slices) => {
                slices.try_for_each(|(start, end)| -> Result<(), ArrowError> {
                    for index in start..end {
                        appended_value_bytes +=
                            Self::append_index(builder, dictionary, values, index)?;
                    }
                    Ok(())
                })?;
            }
            FilterSelection::Indices(indices) => {
                indices.try_for_each(|index| -> Result<(), ArrowError> {
                    appended_value_bytes += Self::append_index(builder, dictionary, values, index)?;
                    Ok(())
                })?;
            }
        }
        self.builder_rows += filter.count();
        self.builder_value_bytes += appended_value_bytes;
        Ok(())
    }

    fn finish(&mut self) -> Result<ArrayRef, ArrowError> {
        self.flush_builder();
        let result = match self.buffered_arrays.len() {
            0 => Arc::new(StringDictionaryBuilder::<Int32Type>::new().finish()) as ArrayRef,
            1 => self.buffered_arrays.pop().unwrap(),
            _ => concat(
                &self
                    .buffered_arrays
                    .iter()
                    .map(|array| array.as_ref())
                    .collect::<Vec<_>>(),
            )?,
        };
        self.buffered_arrays.clear();
        self.total_size_of_non_shared_buffers = 0;
        self.source_data_referenced_in_buffers = false;
        Ok(result)
    }

    fn size(&self) -> usize {
        self.total_size_of_non_shared_buffers
            + self.buffered_arrays.capacity() * size_of::<ArrayRef>()
            + self
                .source
                .as_ref()
                .map_or(0, |source| source.get_array_memory_size())
            + self.builder_rows * size_of::<i32>()
            + self.builder_value_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::FilterBuilder;
    use arrow_array::{BooleanArray, Int32Array};
    use arrow_schema::DataType;

    fn dictionary(values: &[Option<&str>]) -> ArrayRef {
        let mut builder = StringDictionaryBuilder::<Int32Type>::new();
        for value in values {
            match value {
                Some(value) => {
                    builder.append(value).unwrap();
                }
                None => {
                    builder.append_null();
                }
            }
        }
        Arc::new(builder.finish())
    }

    #[test]
    fn preserves_order_across_materialized_and_dense_inputs() {
        let first = dictionary(&[Some("a"), Some("b"), None, Some("a")]);
        let second = dictionary(&[Some("b"), Some("c"), Some("a")]);
        let filter = BooleanArray::from(vec![true, false, true, true]);
        let predicate = FilterBuilder::new(&filter).build();

        let mut in_progress = InProgressStringDictionaryArray::new(8);
        in_progress
            .copy_rows_by_filter_from(Arc::clone(&first), &predicate)
            .unwrap();
        in_progress.set_source(Some(second));
        in_progress.copy_rows(1, 2).unwrap();

        let output = in_progress.finish().unwrap();
        let output = output.as_dictionary::<Int32Type>();
        let values = output.values().as_string::<i32>();
        let decoded = output
            .keys_iter()
            .map(|key| key.map(|key| values.value(key)))
            .collect::<Vec<_>>();
        assert_eq!(
            decoded,
            vec![Some("a"), None, Some("a"), Some("c"), Some("a")]
        );
    }

    #[test]
    fn sparse_copy_unifies_dictionary_values() {
        let input = (0..256)
            .map(|index| match index {
                0 => Some("a"),
                255 => Some("b"),
                _ => Some("unused"),
            })
            .collect::<Vec<_>>();
        let source = dictionary(&input);
        let filter = BooleanArray::from(
            (0..256)
                .map(|index| index == 0 || index == 255)
                .collect::<Vec<_>>(),
        );
        let predicate = FilterBuilder::new(&filter).build();

        let mut in_progress = InProgressStringDictionaryArray::new(8);
        in_progress
            .copy_rows_by_filter_from(source, &predicate)
            .unwrap();
        let output = in_progress.finish().unwrap();
        let output = output.as_dictionary::<Int32Type>();
        let values = output.values().as_string::<i32>();
        let decoded = output
            .keys_iter()
            .map(|key| key.map(|key| values.value(key)))
            .collect::<Vec<_>>();
        assert_eq!(decoded, vec![Some("a"), Some("b")]);
    }

    #[test]
    fn finish_empty_preserves_type() {
        let mut in_progress = InProgressStringDictionaryArray::new(8);
        let output = in_progress.finish().unwrap();
        assert_eq!(
            output.data_type(),
            &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
        );
    }

    #[test]
    fn dictionary_value_nulls_retain_their_representation() {
        let keys = Int32Array::from(vec![0, 1, 0]);
        let values = StringArray::from(vec![Some("a"), None]);
        let source = Arc::new(DictionaryArray::new(keys, Arc::new(values))) as ArrayRef;
        let filter = BooleanArray::from(vec![false, true, true]);
        let predicate = FilterBuilder::new(&filter).build();

        let mut in_progress = InProgressStringDictionaryArray::new(8);
        in_progress
            .copy_rows_by_filter_from(source, &predicate)
            .unwrap();
        let output = in_progress.finish().unwrap();
        let output = output.as_dictionary::<Int32Type>();

        assert!(output.nulls().is_none());
        assert_eq!(output.logical_null_count(), 1);
        assert_eq!(output.keys().values(), &[1, 0]);
    }
}
