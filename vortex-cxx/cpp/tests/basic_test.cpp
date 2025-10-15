// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cstdint>
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <thread>
#include <iostream>
#include <atomic>

#include "vortex/file.hpp"
#include "vortex/scan.hpp"
#include "vortex/write_options.hpp"
#include "vortex/scalar.hpp"
#include "test_data_generator.hpp"
#include "vortex_cxx_bridge/lib.h"

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow.h>

class VortexTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        std::string test_data_path = GetTestDataPath("test_data.vortex");
        auto stream = vortex::testing::CreateTestDataStream();
        auto write_options = vortex::ffi::write_options_new();
        vortex::ffi::write_array_stream(std::move(write_options), reinterpret_cast<uint8_t *>(&stream),
                                        test_data_path.c_str());
    }

protected:
    // Helper function to construct file paths in system temp directory
    static std::string GetTestDataPath(const std::string &filename) {
        std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
        std::filesystem::path vortex_test_dir = temp_dir / "vortex_test";

        if (!std::filesystem::exists(vortex_test_dir)) {
            std::filesystem::create_directories(vortex_test_dir);
        }

        std::filesystem::path target_path = vortex_test_dir / filename;
        return target_path.string();
    }

    // Helper function to create and initialize array view
    nanoarrow::UniqueArrayView CreateArrayView(const nanoarrow::UniqueArray &array,
                                               const nanoarrow::UniqueSchema &schema) {
        nanoarrow::UniqueArrayView array_view;
        ArrowError error;
        ArrowErrorCode init_result = ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), &error);
        if (init_result != NANOARROW_OK) {
            std::cerr << "Error: " << error.message << '\n';
            std::abort();
        }
        ArrowErrorCode set_result = ArrowArrayViewSetArray(array_view.get(), array.get(), nullptr);
        if (set_result != NANOARROW_OK) {
            std::cerr << "Error: " << error.message << '\n';
            std::abort();
        }
        return array_view;
    }

    std::pair<nanoarrow::UniqueArrayStream, nanoarrow::UniqueSchema>
    StreamToUniqueStreamSchema(ArrowArrayStream &stream) {
        nanoarrow::UniqueArrayStream array_stream;
        ArrowArrayStreamMove(&stream, array_stream.get());
        ArrowError error;
        nanoarrow::UniqueSchema schema;
        ArrowErrorCode set_result = ArrowArrayStreamGetSchema(array_stream.get(), schema.get(), &error);
        if (set_result != NANOARROW_OK) {
            std::cerr << "Error: " << error.message << '\n';
            std::abort();
        }
        return {std::move(array_stream), std::move(schema)};
    }

    nanoarrow::UniqueArray ReadFirstArrayFromUniqueStream(nanoarrow::UniqueArrayStream &array_stream) {
        nanoarrow::UniqueArray array;
        int get_next_result = array_stream->get_next(array_stream.get(), array.get());
        EXPECT_EQ(get_next_result, 0);
        return array;
    }

    std::pair<nanoarrow::UniqueArray, nanoarrow::UniqueSchema>
    ReadFirstArrayFromStream(ArrowArrayStream reference_stream) {
        auto [ref_array_stream, ref_schema] = StreamToUniqueStreamSchema(reference_stream);
        auto ref_array = ReadFirstArrayFromUniqueStream(ref_array_stream);
        return {std::move(ref_array), std::move(ref_schema)};
    }

    /// Both array are struct of int64
    void ValidateArray(const nanoarrow::UniqueArray &actual_array,
                       const nanoarrow::UniqueSchema &actual_schema, const nanoarrow::UniqueArray &ref_array,
                       const nanoarrow::UniqueSchema &ref_schema) {
        // Basic properties validation
        ASSERT_EQ(actual_schema->n_children, ref_schema->n_children);

        auto actual_view = CreateArrayView(actual_array, actual_schema);
        auto ref_view = CreateArrayView(ref_array, ref_schema);

        ASSERT_EQ(actual_array->length, ref_array->length);

        // Compare all fields
        for (int64_t field_idx = 0; field_idx < actual_schema->n_children; ++field_idx) {
            auto actual_field = actual_view->children[field_idx];
            auto expected_field = ref_view->children[field_idx];

            ASSERT_EQ(actual_field->array->length, expected_field->array->length);

            for (int64_t i = 0; i < actual_field->array->length; ++i) {
                int64_t actual_value = ArrowArrayViewGetIntUnsafe(actual_field, i);
                int64_t expected_value = ArrowArrayViewGetIntUnsafe(expected_field, i);

                ASSERT_EQ(actual_value, expected_value);
            }
        }
    }

    /// Both array are struct of int64
    void ValidateArrayWithSelection(const nanoarrow::UniqueArray &actual_array,
                                    const nanoarrow::UniqueSchema &actual_schema,
                                    const nanoarrow::UniqueArray &ref_array,
                                    const nanoarrow::UniqueSchema &ref_schema,
                                    const std::vector<int64_t> &row_indices) {
        // Basic properties validation
        ASSERT_EQ(actual_schema->n_children, ref_schema->n_children);
        if (row_indices.empty()) {
            ASSERT_EQ(actual_array->length, 0);
            return;
        }
        auto actual_view = CreateArrayView(actual_array, actual_schema);
        auto ref_view = CreateArrayView(ref_array, ref_schema);

        ASSERT_EQ(actual_array->length, row_indices.size());

        // Selective row comparison using indices
        ASSERT_EQ(actual_array->length, static_cast<int64_t>(row_indices.size()));

        for (int64_t i = 0; i < static_cast<int64_t>(row_indices.size()); ++i) {
            int64_t ref_idx = row_indices[i];

            for (int64_t field_idx = 0; field_idx < actual_schema->n_children; ++field_idx) {
                int64_t actual_val = ArrowArrayViewGetIntUnsafe(actual_view->children[field_idx], i);
                int64_t expected_val = ArrowArrayViewGetIntUnsafe(ref_view->children[field_idx], ref_idx);

                ASSERT_EQ(actual_val, expected_val);
            }
        }
    }

    // Helper to execute scan builder and get array+schema
    std::pair<nanoarrow::UniqueArray, nanoarrow::UniqueSchema> ScanFirstArrayFromTestData(
        const std::function<ArrowArrayStream(vortex::ScanBuilder &)> &configureScanBuilder) {
        auto file = vortex::VortexFile::Open(GetTestDataPath("test_data.vortex"));
        auto scan_builder = file.CreateScanBuilder();
        auto stream = configureScanBuilder(scan_builder);

        auto [array_stream, schema] = StreamToUniqueStreamSchema(stream);
        auto array = ReadFirstArrayFromUniqueStream(array_stream);

        return {std::move(array), std::move(schema)};
    }

    /// Validate array with projection - only checks specified field indices
    void ValidateArrayWithProjection(const nanoarrow::UniqueArray &actual_array,
                                     const nanoarrow::UniqueSchema &actual_schema,
                                     const nanoarrow::UniqueArray &ref_array,
                                     const nanoarrow::UniqueSchema &ref_schema,
                                     const std::vector<int64_t> &field_idxs) {
        ASSERT_EQ(actual_schema->n_children, field_idxs.size());

        auto actual_view = CreateArrayView(actual_array, actual_schema);
        auto ref_view = CreateArrayView(ref_array, ref_schema);

        ASSERT_EQ(actual_array->length, ref_array->length);

        // Compare only the specified fields
        for (int64_t i = 0; i < actual_array->length; ++i) {
            for (size_t proj_idx = 0; proj_idx < field_idxs.size(); ++proj_idx) {
                int64_t ref_field_idx = field_idxs[proj_idx];
                int64_t actual_val = ArrowArrayViewGetIntUnsafe(actual_view->children[proj_idx], i);
                int64_t expected_val = ArrowArrayViewGetIntUnsafe(ref_view->children[ref_field_idx], i);
                ASSERT_EQ(actual_val, expected_val);
            }
        }
    }

    // Top-level test helper that all tests can use
    void
    RunScanBuilderTest(const std::function<ArrowArrayStream(vortex::ScanBuilder &)> &configureScanBuilder,
                       ArrowArrayStream expected_stream,
                       const std::vector<int64_t> &expected_row_indices = {}, bool selection = false) {

        auto [array, schema] = ScanFirstArrayFromTestData(configureScanBuilder);
        auto [ref_array, ref_schema] = ReadFirstArrayFromStream(expected_stream);
        selection == false
            ? ValidateArray(array, schema, ref_array, ref_schema)
            : ValidateArrayWithSelection(array, schema, ref_array, ref_schema, expected_row_indices);
    }

    // New helper for projection tests
    void RunScanBuilderProjectionTest(
        const std::function<ArrowArrayStream(vortex::ScanBuilder &)> &configureScanBuilder,
        ArrowArrayStream expected_stream, const std::vector<int64_t> &field_idxs) {

        auto [array, schema] = ScanFirstArrayFromTestData(configureScanBuilder);
        auto [ref_array, ref_schema] = ReadFirstArrayFromStream(expected_stream);

        ValidateArrayWithProjection(array, schema, ref_array, ref_schema, field_idxs);
    }
};

TEST_F(VortexTest, ScanToStream) {
    RunScanBuilderTest([](vortex::ScanBuilder &builder) { return std::move(builder).IntoStream(); },
                       vortex::testing::CreateTestDataStream());
}

TEST_F(VortexTest, ScanBuilderWithLimitWithRowRange) {
    // Test field "a" and "b" - should contain values from rows 1-2 from original data (indices 1 and
    // 2)
    RunScanBuilderTest(
        [](vortex::ScanBuilder &scan_builder) {
            return std::move(scan_builder.WithLimit(2).WithRowRange(1, 4)).IntoStream();
        },
        vortex::testing::CreateTestDataStream(), {1, 2}, true);
}

TEST_F(VortexTest, ScanBuilderWithIncludeByIndex) {
    std::vector<uint64_t> include_by_index = {1, 3};

    RunScanBuilderTest(
        [&include_by_index](vortex::ScanBuilder &scan_builder) {
            return std::move(
                       scan_builder.WithIncludeByIndex(include_by_index.data(), include_by_index.size()))
                .IntoStream();
        },
        vortex::testing::CreateTestDataStream(), {1, 3}, true);
}

TEST_F(VortexTest, ScanBuilderWithRowRangeWithIncludeByIndex) {
    std::vector<uint64_t> include_by_index = {1, 3, 4};

    RunScanBuilderTest(
        [&include_by_index](vortex::ScanBuilder &scan_builder) {
            return std::move(scan_builder.WithRowRange(2, 6).WithIncludeByIndex(include_by_index.data(),
                                                                                include_by_index.size()))
                .IntoStream();
        },
        vortex::testing::CreateTestDataStream(), {3, 4}, true);
}

TEST_F(VortexTest, WriteArrayStream) {
    auto file = vortex::VortexFile::Open(GetTestDataPath("test_data.vortex"));
    auto stream = file.CreateScanBuilder().IntoStream();

    // Write the stream to a new Vortex file
    vortex::VortexWriteOptions write_options;
    ASSERT_NO_THROW(write_options.WriteArrayStream(stream, GetTestDataPath("test_output.vortex")));

    // Verify the written file
    auto written_file = vortex::VortexFile::Open(GetTestDataPath("test_output.vortex"));
    ASSERT_EQ(written_file.RowCount(), 5);

    // Verify data integrity by reading from the written file
    auto out_stream = written_file.CreateScanBuilder().IntoStream();
    auto [array_stream, schema] = StreamToUniqueStreamSchema(out_stream);
    auto array = ReadFirstArrayFromUniqueStream(array_stream);
    auto [ref_array, ref_schema] = ReadFirstArrayFromStream(vortex::testing::CreateTestDataStream());
    ValidateArray(array, schema, ref_array, ref_schema);
}

TEST_F(VortexTest, ConcurrentMultiStreamRead) {
    std::string test_data_path_1m = GetTestDataPath("test_data_1m.vortex");
    auto stream_1m = vortex::testing::CreateTestData1MStream();
    auto write_options = vortex::ffi::write_options_new();
    vortex::ffi::write_array_stream(std::move(write_options), reinterpret_cast<uint8_t *>(&stream_1m),
                                    test_data_path_1m.c_str());

    auto file = vortex::VortexFile::Open(test_data_path_1m);
    auto stream_driver = file.CreateScanBuilder().IntoStreamDriver();

    // Structure to hold batch data with first ID and nanoarrow array
    struct BatchData {
        int64_t first_id;
        nanoarrow::UniqueArray array;

        BatchData(int64_t first_id, nanoarrow::UniqueArray array)
            : first_id(first_id), array(std::move(array)) {
        }
    };

    std::vector<BatchData> thread1_batches;
    std::vector<BatchData> thread2_batches;

    // Helper function to read from a stream and collect batches
    auto read_stream = [&](std::vector<BatchData> &batches) {
        // Each thread creates its own stream
        auto stream = stream_driver.CreateArrayStream();
        auto [array_stream, schema] = StreamToUniqueStreamSchema(stream);

        std::vector<BatchData> local_batches;

        while (true) {
            nanoarrow::UniqueArray array;
            int get_next_result = array_stream->get_next(array_stream.get(), array.get());

            if (get_next_result != 0) {
                std::cerr << "Error: " << array_stream->get_last_error(array_stream.get()) << '\n';
                std::abort();
            }

            if (array->length == 0) {
                break; // Empty array indicates end
            }

            auto array_view = CreateArrayView(array, schema);

            int64_t first_id = ArrowArrayViewGetIntUnsafe(array_view->children[0], 0);

            local_batches.emplace_back(first_id, std::move(array));
        }
        batches = std::move(local_batches);
    };

    // Launch two threads
    std::thread thread1(read_stream, std::ref(thread1_batches));
    std::thread thread2(read_stream, std::ref(thread2_batches));

    // Wait for both threads to complete
    thread1.join();
    thread2.join();

    // Combine all batches from both threads
    std::vector<BatchData> all_batches;
    all_batches.insert(all_batches.end(), std::make_move_iterator(thread1_batches.begin()),
                       std::make_move_iterator(thread1_batches.end()));
    all_batches.insert(all_batches.end(), std::make_move_iterator(thread2_batches.begin()),
                       std::make_move_iterator(thread2_batches.end()));

    // Sort batches by first ID to ensure proper validation order
    std::sort(all_batches.begin(), all_batches.end(),
              [](const BatchData &a, const BatchData &b) { return a.first_id < b.first_id; });

    // Create reference data for validation
    auto [ref_array, ref_schema] = ReadFirstArrayFromStream(vortex::testing::CreateTestData1MStream());
    auto ref_array_view = CreateArrayView(ref_array, ref_schema);

    // Validate all data against reference
    constexpr size_t EXPECTED_ROWS = static_cast<size_t>(1024) * 1024;
    size_t total_rows_read = 0;
    int64_t reference_offset = 0;

    auto stream_for_schema = stream_driver.CreateArrayStream();
    auto [_, schema] = StreamToUniqueStreamSchema(stream_for_schema);
    for (const auto &batch : all_batches) {
        auto array_view = CreateArrayView(batch.array, schema);

        for (int64_t i = 0; i < batch.array->length; ++i) {

            int64_t actual_id = ArrowArrayViewGetIntUnsafe(array_view->children[0], i);
            int32_t actual_value =
                static_cast<int32_t>(ArrowArrayViewGetIntUnsafe(array_view->children[1], i));

            int64_t expected_id = ArrowArrayViewGetIntUnsafe(ref_array_view->children[0], reference_offset);
            int32_t expected_value = static_cast<int32_t>(
                ArrowArrayViewGetIntUnsafe(ref_array_view->children[1], reference_offset));

            ASSERT_EQ(actual_id, expected_id);
            ASSERT_EQ(actual_value, expected_value);
            reference_offset++;
        }
        total_rows_read += batch.array->length;
    }

    // Verify we read all expected data
    ASSERT_EQ(total_rows_read, EXPECTED_ROWS)
        << "Expected to read " << EXPECTED_ROWS << " rows, but read " << total_rows_read << " rows";
    ASSERT_EQ(reference_offset, EXPECTED_ROWS) << "Reference validation didn't cover all rows";

    ASSERT_GT(all_batches.size(), 1) << "Expected multiple batches, but got " << all_batches.size();
}

namespace ve = vortex::expr;
namespace vs = vortex::scalar;

TEST_F(VortexTest, ScanBuilderWithFilter) {
    // Test filtering with eq(column("a"), val) - should return only rows where column "a" equals 30
    RunScanBuilderTest(
        [](vortex::ScanBuilder &scan_builder) {
            auto filter = ve::eq(ve::column("a"), ve::literal(vs::int32(30)));
            return std::move(scan_builder.WithFilter(std::move(filter))).IntoStream();
        },
        vortex::testing::CreateTestDataStream(), {2}, true); // Row index 2 corresponds to value 30
}

TEST_F(VortexTest, ScanBuilderWithFilterLvalueref) {
    // Test filtering with eq(column("a"), val) - should return only rows where column "a" equals 30
    RunScanBuilderTest(
        [](vortex::ScanBuilder &scan_builder) {
            const auto filter = ve::eq(ve::column("a"), ve::literal(vs::int32(30)));
            return std::move(scan_builder.WithFilter(filter)).IntoStream();
        },
        vortex::testing::CreateTestDataStream(), {2}, true); // Row index 2 corresponds to value 30
}

TEST_F(VortexTest, ScanBuilderWithFilterNoMatches) {
    // Test filtering with eq(column("a"), val) where no rows match - should return empty result
    RunScanBuilderTest(
        [](vortex::ScanBuilder &scan_builder) {
            auto filter = ve::eq(ve::column("a"),
                                 ve::literal(vs::int32(999)) // Value that doesn't exist in test data
            );
            return std::move(scan_builder.WithFilter(std::move(filter))).IntoStream();
        },
        vortex::testing::CreateTestDataStream(), {}, true); // No matching rows
}

TEST_F(VortexTest, ScanBuilderWithFilterUsingDTypeFromArrowAndScalarCast) {
    // Test filtering using DType::from_arrow and Scalar::cast functionality
    // This test creates a filter expression by casting a scalar to match the column type

    // Test DType::from_arrow with int32 schema
    nanoarrow::UniqueSchema int32_schema;
    ArrowSchemaInit(int32_schema.get());
    ArrowErrorCode result = ArrowSchemaSetType(int32_schema.get(), NANOARROW_TYPE_INT32);
    EXPECT_EQ(result, NANOARROW_OK);
    result = ArrowSchemaSetName(int32_schema.get(), "test_field");
    EXPECT_EQ(result, NANOARROW_OK);

    auto dtype = vortex::dtype::from_arrow(*int32_schema.get());

    // Use the casted scalar in filter expression - create a new scalar for lambda
    RunScanBuilderTest(
        [&](vortex::ScanBuilder &scan_builder) {
            auto test_scalar = vs::cast(vs::int64(30), std::move(dtype));
            auto filter = ve::eq(ve::column("a"), ve::literal(std::move(test_scalar)));
            return std::move(scan_builder.WithFilter(std::move(filter))).IntoStream();
        },
        vortex::testing::CreateTestDataStream(), {2}, true); // Row index 2 corresponds to value 30
}

TEST_F(VortexTest, ScanBuilderWithProjectionSingleColumn) {
    // Test projection selecting only column "a" (field index 0)
    RunScanBuilderProjectionTest(
        [](vortex::ScanBuilder &scan_builder) {
            auto projection = ve::select({"a"}, ve::root());
            return std::move(scan_builder.WithProjection(std::move(projection))).IntoStream();
        },
        vortex::testing::CreateTestDataStream(), {0});
}

TEST_F(VortexTest, OpenFromBuffer) {
    std::string test_file_path = GetTestDataPath("test_buffer.vortex");
    auto stream = vortex::testing::CreateTestDataStream();
    auto write_options = vortex::ffi::write_options_new();
    vortex::ffi::write_array_stream(std::move(write_options), reinterpret_cast<uint8_t *>(&stream),
                                    test_file_path.c_str());

    std::ifstream file(test_file_path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(file.is_open()) << "Failed to open file: " << test_file_path;

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(file_size);
    ASSERT_TRUE(file.read(reinterpret_cast<char*>(buffer.data()), file_size))
        << "Failed to read file into buffer";
    file.close();

    auto vortex_file = vortex::VortexFile::Open(buffer.data(), buffer.size());
    ASSERT_EQ(vortex_file.RowCount(), 5);

    auto scan_stream = vortex_file.CreateScanBuilder().IntoStream();
    auto [array_stream, schema] = StreamToUniqueStreamSchema(scan_stream);
    auto array = ReadFirstArrayFromUniqueStream(array_stream);

    auto [ref_array, ref_schema] = ReadFirstArrayFromStream(vortex::testing::CreateTestDataStream());
    ValidateArray(array, schema, ref_array, ref_schema);
}

class TrackingReadAt : public vortex::io::VortexReadAt {
private:
    std::string file_path_;
    mutable std::atomic<size_t> bytes_read_{0};
    mutable std::atomic<size_t> read_count_{0};

public:
    explicit TrackingReadAt(std::string file_path) : file_path_(std::move(file_path)) {}

    std::vector<uint8_t> ReadAt(uint64_t pos, size_t len) const override {
        std::ifstream file(file_path_, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path_);
        }

        file.seekg(pos);
        std::vector<uint8_t> data(len);
        file.read(reinterpret_cast<char*>(data.data()), len);

        size_t actually_read = file.gcount();
        bytes_read_.fetch_add(actually_read, std::memory_order_relaxed);
        read_count_.fetch_add(1, std::memory_order_relaxed);

        data.resize(actually_read);
        return data;
    }

    uint64_t GetSize() const override {
        std::ifstream file(file_path_, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path_);
        }
        return file.tellg();
    }

    size_t GetBytesRead() const { return bytes_read_.load(std::memory_order_relaxed); }
    size_t GetReadCount() const { return read_count_.load(std::memory_order_relaxed); }
    void ResetCounters() {
        bytes_read_.store(0, std::memory_order_relaxed);
        read_count_.store(0, std::memory_order_relaxed);
    }
};

TEST_F(VortexTest, OpenWithReadAtMetadataOnly) {
    std::string large_file_path = GetTestDataPath("test_data_large.vortex");
    constexpr size_t NUM_ROWS = 20'000;
    auto large_stream = vortex::testing::CreateRandomDataStream(NUM_ROWS);
    auto write_options = vortex::ffi::write_options_new();
    vortex::ffi::write_array_stream(std::move(write_options),
                                    reinterpret_cast<uint8_t*>(&large_stream),
                                    large_file_path.c_str());

    std::ifstream file(large_file_path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(file.is_open()) << "Failed to open file: " << large_file_path;
    uint64_t file_size = file.tellg();
    file.close();

    // File must be larger than minimum footer read size
    constexpr size_t MIN_FOOTER_READ = 65'535;
    ASSERT_GT(file_size, MIN_FOOTER_READ)
        << "Test file must be > " << MIN_FOOTER_READ << " bytes, got " << file_size;

    auto tracking_reader = std::make_unique<TrackingReadAt>(large_file_path);
    auto *reader_ptr = tracking_reader.get();
    auto vortex_file = vortex::VortexFile::OpenSeekable(std::move(tracking_reader));

    // Verify that we only read a small portion (metadata from the end)
    size_t bytes_read = reader_ptr->GetBytesRead();
    size_t read_count = reader_ptr->GetReadCount();

    std::cout << "Bytes read during open: " << bytes_read << ", read operations: " << read_count
              << ", file size: " << file_size << '\n';

    ASSERT_LT(bytes_read, file_size)
        << "Should not read entire file during open";
    ASSERT_EQ(read_count, 1)
        << "Should perform only one read operation to fetch metadata";

    // Verify the file is usable and we got the metadata correctly
    ASSERT_EQ(vortex_file.RowCount(), NUM_ROWS);
}
