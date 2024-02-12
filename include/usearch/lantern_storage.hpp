
#pragma once

#include <cstdio>
#include <exception>
#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>
#include <usearch/storage.hpp>

#ifndef NDEBUG
#include <iostream>
#include <string>
#endif // !NDEBUG

namespace unum {
namespace usearch {

/**
 * @brief  A simple Storage implementation that uses standard cpp containers and complies with the usearch storage
 *abstraction for HNSW graph and associated vector data
 *
 *  @tparam key_at
 *      The type of primary objects stored in the index.
 *      The values, to which those map, are not managed by the same index structure.
 *
 *  @tparam compressed_slot_at
 *      The smallest unsigned integer type to address indexed elements.
 *      It is used internally to maximize space-efficiency and is generally
 *      up-casted to @b `std::size_t` in public interfaces.
 *      Can be a built-in @b `uint32_t`, `uint64_t`, or our custom @b `uint40_t`.
 *      Which makes the most sense for 4B+ entry indexes.
 *
 *  @tparam allocator_at
 *      Potentially different memory allocator for primary allocations of nodes and vectors.
 *      The allocated buffers may be uninitialized.
 *      Note that we are using a memory aaligned allocator in place of std::allocator<byte_t>
 *      Because of scalar_t memory requirements in index_*
 *
 **/
template <bool is_external_ak, typename key_at, typename compressed_slot_at,
          typename allocator_at = aligned_allocator_gt<byte_t, 64>> //
class lantern_storage_gt {
  public:
    using key_t = key_at;
    using node_t = node_at<key_t, compressed_slot_at>;
    using span_bytes_t = span_gt<byte_t>;

  private:
    using dynamic_allocator_traits_t = std::allocator_traits<dynamic_allocator_t>;
    using levels_allocator_t = typename dynamic_allocator_traits_t::template rebind_alloc<level_t>;
    using nodes_allocator_t = typename dynamic_allocator_traits_t::template rebind_alloc<node_t>;
    using offsets_allocator_t = typename dynamic_allocator_traits_t::template rebind_alloc<std::size_t>;
    using vectors_allocator_t = typename dynamic_allocator_traits_t::template rebind_alloc<byte_t*>;

    using node_retriever_t = void* (*)(void* ctx, int index);

    using nodes_mutexes_t = bitset_gt<dynamic_allocator_t>;
    using nodes_t = buffer_gt<node_t, nodes_allocator_t>;
    using vectors_t = std::vector<span_bytes_t>;

    nodes_t nodes_{};
    vectors_t vectors_{};
    vectors_t vectors_pq_{};
    mutable nodes_mutexes_t nodes_mutexes_{};

    node_retriever_t external_node_retriever_{};
    node_retriever_t external_node_retriever_mut_{};
    void* retriever_ctx_{};

    precomputed_constants_t pre_{};
    allocator_at allocator_{};
    static_assert(!has_reset<allocator_at>(), "reset()-able memory allocators not supported for this storage provider");
    memory_mapped_file_t viewed_file_{};
    // the next three are used only in serialization/deserialization routines to know how to serialize vectors
    // since this is only for serde/vars are marked mutable to still allow const-ness of saving method interface on
    // storage instance
    mutable size_t node_count_{};
    bool loaded_ = false;
    bool pq_{};
    float* pq_codebook_{};
    byte_t* pq_constant_{};
    mutable size_t vector_size_{};
    // defaulted to true because that is what test.cpp assumes when using this storage directly
    mutable bool exclude_vectors_ = true;
    // used to maintain proper alignment in stored indexes to make sure view() does not result in misaligned accesses
    mutable size_t file_offset_{};

    // used in place of error handling throughout the class
    static void expect(bool must_be_true) {
        if (must_be_true)
            return;
        if constexpr (is_external_ak) {
            // no good way to get out of here, or even log, since caller may be in postgres
            // the fprintf will appear in server logs
            fprintf(stderr, "LANTERN STORAGE: unexpected invariant violation in storage layer");
            std::terminate();
        } else {

            throw std::runtime_error("LANTERN STORAGE: unexpected invariant violation in storage layer");
        }
    }
    // padding buffer, some prefix of which will be used every time we need padding in the serialization
    // of the index.
    // Rest of the array will be zeros but we will also never need paddings that large
    // The pattern is to help in debugging
    // Note: 'inline' is crucial to let this compile into C environments (why?)
    // otherwise symbol _ZN4unum7usearch14std_storage_atImjNS0_20aligned_allocator_gtIcLm64EEEE14padding_bufferE
    // leaks out
    constexpr static inline byte_t padding_buffer[64] = {0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42};

    template <typename A, typename T> size_t align(T v) const {
        return (sizeof(A) - (size_t)v % sizeof(A)) % sizeof(A);
    }

    template <typename T> size_t align4(T) const {
        // todo:: fix alignment issues in lantern;
        return 0;
        // return align<float>(v);
    }

  public:
    lantern_storage_gt(index_config_t config, allocator_at allocator = {})
        : pre_(node_t::precompute_(config)), allocator_(allocator), pq_(config.pq) {
        if (pq_) {

            pq_constant_ = allocator_.allocate(vector_size_ * 256);
            expect(pq_constant_);
        }
        memset(pq_constant_, 0, vector_size_ * 256);
    }
    lantern_storage_gt(index_config_t config, float* codebook, allocator_at allocator = {})
        : pre_(node_t::precompute_(config)), allocator_(allocator), pq_(config.pq) {
        // if (codebook)
        //     assert(pq_ == true);
        pq_constant_ = allocator_.allocate(vector_size_ * 256);
        expect(pq_constant_);
        memset(pq_constant_, 0, vector_size_ * 256);

        pq_codebook_ = codebook;
    }

    inline node_t get_node_at(std::size_t idx) const noexcept {
        // std::cerr << "getting node at" << std::to_string(idx) << std::endl;
        if (loaded_ && is_external_ak) {
            assert(retriever_ctx_ != nullptr);
            char* tape = (char*)external_node_retriever_(retriever_ctx_, idx);
            return node_t{tape};
        }

        return nodes_[idx];
    }
    inline byte_t* get_vector_at(std::size_t idx) const noexcept {
        if (loaded_ && is_external_ak) {
            assert(retriever_ctx_ != nullptr);
            char* tape = (char*)external_node_retriever_(retriever_ctx_, idx);
            if (pq_)
                return pq_constant_;
            node_t node{tape};
            return tape + node.node_size_bytes(pre_);
        }
        if (pq_) {
            // todo:: do decompression here
            return pq_constant_;
        }

        return vectors_[idx].data();
    }
    inline size_t node_size_bytes(std::size_t idx) const noexcept { return get_node_at(idx).node_size_bytes(pre_); }
    bool is_immutable() const noexcept { return bool(viewed_file_); }

    void set_node_retriever(void* retriever_ctx, node_retriever_t external_node_retriever,
                            node_retriever_t external_node_retriever_mut) noexcept {
        retriever_ctx_ = retriever_ctx;
        external_node_retriever_ = external_node_retriever;
        external_node_retriever_mut_ = external_node_retriever_mut;
    }

    struct dummy_lock {
        // destructor necessary to avoid "unused variable warning"
        // at callcites of node_lock()
        ~dummy_lock() = default;
    };

    struct node_lock_t {
        nodes_mutexes_t& mutexes;
        std::size_t slot;
        inline ~node_lock_t() noexcept { mutexes.atomic_reset(slot); }
    };

    // when using external storage, the external storage is responsible for doing appropriate locking before passing
    // objects to us, so we use a dummy lock here When allocating nodes ourselves, however, we do proper per=node
    // locking with a bitfield, identical to how upstream usearch storeage does it
    using lock_type = std::conditional_t<is_external_ak, dummy_lock, node_lock_t>;

    bool reserve(std::size_t count) {
        if (loaded_ && retriever_ctx_ != nullptr && is_external_ak) {
            // we will be using external storage, no need to reserve
            return true;
        }

        if (count < nodes_.size() && count < nodes_mutexes_.size())
            return true;
        nodes_mutexes_t new_mutexes(count);
        nodes_t new_nodes(count);
        if (!new_mutexes || !new_nodes)
            return false;
        if (nodes_)
            std::memcpy(new_nodes.data(), nodes_.data(), sizeof(node_t) * nodes_.size());

        nodes_mutexes_ = std::move(new_mutexes);
        nodes_ = std::move(new_nodes);
        vectors_.resize(count);
        if (pq_)
            vectors_pq_.resize(count);

        return true;
    }
    void clear() noexcept {
        if (!is_immutable()) {
            std::size_t n = nodes_.size();
            for (std::size_t i = 0; i != n; ++i) {
                // we do not know which slots have been filled and which ones - no
                // so we iterate over full reserved space
                if (nodes_[i])
                    node_free(i, nodes_[i]);
            }
            n = vectors_.size();
            for (std::size_t i = 0; i != n; ++i) {
                span_bytes_t v = vectors_[i];
                if (v.data()) {
                    allocator_.deallocate(v.data(), v.size());
                }
            }
            n = vectors_pq_.size();
            for (std::size_t i = 0; i != n; ++i) {
                span_bytes_t v = vectors_pq_[i];
                if (v.data()) {
                    allocator_.deallocate(v.data(), v.size());
                }
            }
        }
        if (vectors_.data())
            std::fill(vectors_.begin(), vectors_.end(), span_bytes_t{});
        if (vectors_pq_.data())
            std::fill(vectors_pq_.begin(), vectors_pq_.end(), span_bytes_t{});
        if (nodes_.data())
            std::fill(nodes_.begin(), nodes_.end(), node_t{});
    }
    void reset() noexcept { clear(); }

    span_bytes_t node_malloc(level_t level) noexcept {
        std::size_t node_size = node_t::node_size_bytes(pre_, level);
        byte_t* data = (byte_t*)allocator_.allocate(node_size);
        return data ? span_bytes_t{data, node_size} : span_bytes_t{};
    }
    void node_free(size_t slot, node_t node) {
        allocator_.deallocate(node.tape(), node.node_size_bytes(pre_));
        nodes_[slot] = node_t{};
    }
    node_t node_make(key_at key, level_t level) noexcept {
        if (loaded_ && is_external_ak)
            return node_t{(char*)0x42};
        span_bytes_t node_bytes = node_malloc(level);
        if (!node_bytes)
            return {};

        std::memset(node_bytes.data(), 0, node_bytes.size());
        node_t node{(byte_t*)node_bytes.data()};
        node.key(key);
        node.level(level);
        return node;
    }
    void node_store(size_t slot, node_t node) noexcept {
        if (loaded_ && is_external_ak)
            return;
        nodes_[slot] = node;
    }
    void set_vector_at(size_t slot, const byte_t* vector_data, size_t vector_size, bool copy_vector, bool reuse_node) {
        if (loaded_ && is_external_ak)
            return;

        usearch_assert_m(!(reuse_node && !copy_vector),
                         "Cannot reuse node when not copying as there is no allocation needed");

        const int pq_size = vector_size / 8 / 4;
        if (copy_vector) {
            if (!reuse_node) {
                vectors_[slot] = span_bytes_t{allocator_.allocate(vector_size), vector_size};
                if (pq_) {
                    vectors_pq_[slot] = span_bytes_t{allocator_.allocate(pq_size), pq_size};
                }
            }
            std::memcpy(vectors_[slot].data(), vector_data, vector_size);
            if (pq_)
                std::memcpy(vectors_pq_[slot].data(), vector_data + vector_size, pq_size);

            std::cerr << "the 2 chars after vector: " << std::to_string(*(char*)(vector_data + vector_size)) << " "
                      << std::to_string(*(char*)(vector_data + vector_size + 1)) << std::endl;
        } else {
            vectors_[slot] = span_bytes_t{(byte_t*)vector_data, vector_size};
            if (pq_)
                vectors_pq_[slot] = span_bytes_t{(byte_t*)vector_data + vector_size, pq_size};
        }
    }

    allocator_at const& node_allocator() const noexcept { return allocator_; }

    constexpr inline lock_type node_lock(std::size_t i) const noexcept {
        if constexpr (is_external_ak) {
            return {};
        } else {
            while (nodes_mutexes_.atomic_set(i))
                ;
            return {nodes_mutexes_, i};
        }
    }

    // serialization

    template <typename output_callback_at, typename vectors_metadata_at>
    serialization_result_t save_vectors_to_stream(output_callback_at& output, std::uint64_t vector_size_bytes,
                                                  std::uint64_t node_count, //
                                                  const vectors_metadata_at& metadata_buffer,
                                                  serialization_config_t config = {}) const {
        expect(!config.use_64_bit_dimensions);
        expect(output(metadata_buffer, sizeof(metadata_buffer)));

        file_offset_ = sizeof(metadata_buffer);
        vector_size_ = vector_size_bytes;
        node_count_ = node_count;
        exclude_vectors_ = config.exclude_vectors;
        return {};
    }

    template <typename output_callback_at, typename progress_at = dummy_progress_t>
    serialization_result_t save_nodes_to_stream(output_callback_at& output, const index_serialized_header_t& header,
                                                progress_at& = {}) const {
        expect(output(&header, sizeof(header)));
        expect(output(&vector_size_, sizeof(vector_size_)));
        expect(output(&node_count_, sizeof(node_count_)));
        file_offset_ += sizeof(header) + sizeof(vector_size_) + sizeof(node_count_);
        if (loaded_ && file_offset_ >= 136 && is_external_ak) {
            return {};
        }

        // N.B: unlike upstream usearch storage, lantern storage does not save level info of all nodes as part of the
        // header This in upstream storage speeds up view()-ing index from disc, but that API is not relevant in lantern
        // and having level info here unnecessarily bloats our index, so we do not do it.

        // After that dump the nodes themselves
        for (std::size_t i = 0; i != header.size; ++i) {
            span_bytes_t node_bytes = get_node_at(i).node_bytes(pre_);
            expect(output(node_bytes.data(), node_bytes.size()));
            // std::fprintf(stderr, "node %d level %d size %d offset %d\n", (int)i, (int)get_node_at(i).level(),
            //              (int)node_bytes.size(), (int)file_offset_);
            file_offset_ += node_bytes.size();

            if (!exclude_vectors_) {
                // add padding for proper alignment
                int16_t padding_size = align4(file_offset_);
                expect(output(&padding_buffer, padding_size));
                file_offset_ += padding_size;
                byte_t* vector_bytes = get_vector_at(i);
                expect(output(vector_bytes, vector_size_));
                file_offset_ += vector_size_;
            }
        }
        return {};
    }

    template <typename input_callback_at, typename vectors_metadata_at>
    serialization_result_t load_vectors_from_stream(input_callback_at& input, //
                                                    vectors_metadata_at& metadata_buffer,
                                                    serialization_config_t config = {}, bool = false) {
        expect(!config.use_64_bit_dimensions);
        expect(input(metadata_buffer, sizeof(metadata_buffer)));
        file_offset_ = sizeof(metadata_buffer);
        exclude_vectors_ = config.exclude_vectors;
        return {};
    }

    template <typename input_callback_at, typename progress_at = dummy_progress_t>
    serialization_result_t load_nodes_from_stream(input_callback_at& input, index_serialized_header_t& header,
                                                  progress_at& = {}) noexcept {
        expect(input(&header, sizeof(header)));
        expect(input(&vector_size_, sizeof(vector_size_)));
        expect(input(&node_count_, sizeof(node_count_)));
        loaded_ = true;
        file_offset_ += sizeof(header) + sizeof(vector_size_) + sizeof(node_count_);
        if (!header.size) {
            reset();
            return {};
        }
        if (external_node_retriever_ && is_external_ak)
            return {};
        byte_t in_padding_buffer[64] = {0};

        expect(reserve(header.size));

        // N.B: unlike upstream usearch storage, lantern storage does not save level info of all nodes as part of the
        // header This in upstream storage speeds up view()-ing index from disc, but that API is not relevant in lantern
        // and having level info here unnecessarily bloats our index, so we do not do it.

        std::array<byte_t, node_t::head_size_bytes()> node_header;
        level_t extracted_node_level = -1;

        // Load the nodes
        for (std::size_t i = 0; i != header.size; ++i) {
            // extract just the node header first, to know its level, then extract the rest
            expect(input(node_header.data(), node_header.size()));
            extracted_node_level = node_t{node_header.data()}.level();
            span_bytes_t node_bytes = node_malloc(extracted_node_level);

            std::memcpy(node_bytes.data(), node_header.data(), node_t::head_size_bytes());
            expect(input(node_bytes.data() + node_t::head_size_bytes(), //
                         node_bytes.size() - node_t::head_size_bytes()));

            file_offset_ += node_bytes.size();
            node_store(i, node_t{node_bytes.data()});
            if (!exclude_vectors_) {
                int16_t padding_size = align4(file_offset_);
                expect(input(&in_padding_buffer, padding_size));
                file_offset_ += padding_size;
                expect(std::memcmp(in_padding_buffer, padding_buffer, padding_size) == 0);
                byte_t* vector_bytes = allocator_.allocate(vector_size_);
                expect(input(vector_bytes, vector_size_));
                file_offset_ += vector_size_;
                set_vector_at(i, vector_bytes, vector_size_, false, false);
            }
        }
        return {};
    }

    template <typename vectors_metadata_at>
    serialization_result_t view_vectors_from_file(
        memory_mapped_file_t& file, //
                                    //// todo!! document that offset is a reference, or better - do not do it this way
        vectors_metadata_at& metadata_buffer, std::size_t& offset, serialization_config_t config = {}) {
        if constexpr (is_external_ak) {
            return serialization_result_t{}.failed("cannot view vectors when storage is external");
        }
        reset();
        exclude_vectors_ = config.exclude_vectors;
        expect(!config.use_64_bit_dimensions);

        expect(bool(file.open_if_not()));
        std::memcpy(metadata_buffer, file.data() + offset, sizeof(metadata_buffer));
        file_offset_ = sizeof(metadata_buffer);
        offset += sizeof(metadata_buffer);
        return {};
    }

    template <typename progress_at = dummy_progress_t>
    serialization_result_t view_nodes_from_file(memory_mapped_file_t file, index_serialized_header_t& header,
                                                std::size_t offset = 0, progress_at& = {}) noexcept {
        if constexpr (is_external_ak) {
            return serialization_result_t{}.failed("cannot view vectors when storage is external");
        }
        serialization_result_t result = file.open_if_not();
        std::memcpy(&header, file.data() + offset, sizeof(header));
        offset += sizeof(header);
        std::memcpy(&vector_size_, file.data() + offset, sizeof(vector_size_));
        offset += sizeof(vector_size_);
        std::memcpy(&node_count_, file.data() + offset, sizeof(node_count_));
        offset += sizeof(node_count_);
        if (!header.size) {
            reset();
            return result;
        }
        index_config_t config;
        config.connectivity = header.connectivity;
        config.connectivity_base = header.connectivity_base;
        pre_ = node_t::precompute_(config);
        buffer_gt<std::size_t> offsets(header.size);
        expect(offsets);

        expect(reserve(header.size));
        // N.B: unlike upstream usearch storage, lantern storage does not save level info of all nodes as part of the
        // header This in upstream storage speeds up view()-ing index from disc, but that API is not relevant in lantern
        // and having level info here unnecessarily bloats our index, so we do not do it.

        offsets[0u] = offset;

        // Rapidly address all the nodes and vectors
        for (std::size_t i = 0; i != header.size; ++i) {
            // offset of 0th is already above, offset of each next one comes from the node before
            if (i > 0) {
                offsets[i] = offsets[i - 1] + get_node_at(i - 1).node_size_bytes(pre_);
                // node_t::node_size_bytes(pre_, levels[i - 1]);
                if (!exclude_vectors_) {
                    // add room for vector alignment
                    offsets[i] += align4(offsets[i]);
                    offsets[i] += vector_size_;
                }
            }
            node_store(i, node_t{file.data() + offsets[i]});

            if (!exclude_vectors_) {
                size_t vector_offset = offsets[i] + node_size_bytes(i);
                expect(std::memcmp(file.data() + vector_offset, padding_buffer, align4(vector_offset)) == 0);
                vector_offset += align4(vector_offset);

                // expect proper alignment
                expect(align4(vector_offset) == 0);
                expect(align4(file.data() + vector_offset) == 0);
                set_vector_at(i, file.data() + vector_offset, vector_size_, false, false);
            }
        }
        viewed_file_ = std::move(file);
        return {};
    }
};

template <typename key_at, typename compressed_slot_at, typename allocator_at = aligned_allocator_gt<byte_t, 64>>
using lantern_storage_at = lantern_storage_gt<false, key_at, compressed_slot_at, allocator_at>;

using lantern_external_storage_t = lantern_storage_gt<true, default_key_t, default_slot_t>;
using lantern_internal_storage_t = lantern_storage_gt<false, default_key_t, default_slot_t>;

ASSERT_VALID_STORAGE(lantern_external_storage_t);
ASSERT_VALID_STORAGE(lantern_internal_storage_t);

} // namespace usearch
} // namespace unum
