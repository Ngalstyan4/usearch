#pragma once
#include <stdlib.h> // `aligned_alloc`

#include <functional>   // `std::function`
#include <numeric>      // `std::iota`
#include <shared_mutex> // `std::shared_mutex`
#include <thread>       // `std::thread`
#include <vector>       // `std::vector`

#include <usearch/index.hpp>
#include <usearch/index_punned_helpers.hpp>

#include <tsl/robin_map.h>

namespace unum {
namespace usearch {

struct cos_f8_t {
    using scalar_t = f8_bits_t;
    std::size_t dimensions;

    inline cos_f8_t(std::size_t dims) noexcept : dimensions(dims) {}
    inline metric_kind_t kind() const noexcept { return metric_kind_t::cos_k; }
    inline punned_distance_t operator()(f8_bits_t const* a, f8_bits_t const* b) const noexcept {
        std::int32_t ab{}, a2{}, b2{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : ab, a2, b2)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != dimensions; i++) {
            std::int16_t ai{a[i]};
            std::int16_t bi{b[i]};
            ab += ai * bi;
            a2 += square(ai);
            b2 += square(bi);
        }
        return (ab != 0) ? (1.f - ab / (std::sqrt(a2) * std::sqrt(b2))) : 0;
    }
};

struct l2sq_f8_t {
    using scalar_t = f8_bits_t;
    std::size_t dimensions;

    inline l2sq_f8_t(std::size_t dims) noexcept : dimensions(dims) {}
    inline metric_kind_t kind() const noexcept { return metric_kind_t::l2sq_k; }
    inline scalar_kind_t scalar_kind() const noexcept { return scalar_kind_t::f8_k; }
    inline punned_distance_t operator()(f8_bits_t const* a, f8_bits_t const* b) const noexcept {
        std::int32_t ab_deltas_sq{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : ab_deltas_sq)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != dimensions; i++)
            ab_deltas_sq += square(std::int16_t(a[i]) - std::int16_t(b[i]));
        return ab_deltas_sq;
    }
};

struct index_punned_dense_metric_t {
    using scalar_t = byte_t;
    using result_t = punned_distance_t;
    using view_t = punned_vector_view_t;
    using stl_func_t = std::function<punned_distance_t(punned_vector_view_t, punned_vector_view_t)>;

    stl_func_t func_;
    metric_kind_t kind_ = metric_kind_t::unknown_k;
    scalar_kind_t scalar_kind_ = scalar_kind_t::unknown_k;
    isa_t isa_ = isa_t::auto_k;

    index_punned_dense_metric_t() = default;

    template <typename typed_at, typename scalar_at = typename typed_at::scalar_t>
    index_punned_dense_metric_t(typed_at typed)
        : index_punned_dense_metric_t(typed.kind(), isa_t::auto_k, scalar_at{}, typed) {}

    template <typename typed_at, typename scalar_at>
    index_punned_dense_metric_t(metric_kind_t kind, isa_t isa, scalar_at, typed_at metric) {
        using scalar_t = scalar_at;
        using typed_view_t = span_gt<scalar_t const>;
        func_ = [=](view_t a, view_t b) -> result_t {
            std::size_t dims = a.size() / sizeof(scalar_t);
            typed_view_t a_typed{(scalar_t const*)a.data(), dims};
            typed_view_t b_typed{(scalar_t const*)b.data(), dims};
            return metric(a_typed, b_typed);
        };
        if (std::is_same<scalar_at, f8_bits_t>())
            scalar_kind_ = scalar_kind_t::f8_k;
        else if (std::is_same<scalar_at, f16_bits_t>())
            scalar_kind_ = scalar_kind_t::f16_k;
        else
            scalar_kind_ = common_scalar_kind<scalar_at>();
        kind_ = kind;
        isa_ = isa;
    }

    inline metric_kind_t kind() const noexcept { return kind_; }
    inline scalar_kind_t scalar_kind() const noexcept { return scalar_kind_; }
    inline result_t operator()(view_t a, view_t b) const { return func_(a, b); }
};

constexpr std::size_t default_removals_cycle() { return 64; }

template <typename label_at, typename std::enable_if<std::is_integral<label_at>::value>::type* = nullptr>
label_at default_free_value() {
    return std::numeric_limits<label_at>::max();
}

template <typename label_at, typename std::enable_if<std::is_same<label_at, uint40_t>::value>::type* = nullptr>
uint40_t default_free_value() {
    return uint40_t::max();
}

template <                                                        //
    typename label_at,                                            //
    typename std::enable_if<!std::is_integral<label_at>::value && //
                            !std::is_same<label_at, uint40_t>::value>::type* = nullptr>
label_at default_free_value() {
    return label_at();
}

/**
 *  @brief  Oversimplified type-punned index for equidimensional vectors
 *          with automatic @b down-casting, hardware-specific @b SIMD metrics,
 *          and ability to @b remove existing vectors, common in Semantic Caching
 *          applications.
 */
template <typename label_at = std::int64_t, typename id_at = std::uint32_t> //
class index_punned_dense_gt {
  public:
    using label_t = label_at;
    using id_t = id_at;
    using distance_t = punned_distance_t;
    /// @brief Punned metric object.
    using metric_t = index_punned_dense_metric_t;

  private:
    /// @brief Schema: input buffer, bytes in input buffer, output buffer.
    using cast_t = std::function<bool(byte_t const*, std::size_t, byte_t*)>;
    /// @brief Punned index.
    using index_t = index_gt<metric_t, label_t, id_t, aligned_allocator_t, memory_mapping_allocator_t>;
    using index_allocator_t = aligned_allocator_gt<index_t, 64>;

    using member_iterator_t = typename index_t::member_iterator_t;
    using member_citerator_t = typename index_t::member_citerator_t;
    using member_ref_t = typename index_t::member_ref_t;
    using member_cref_t = typename index_t::member_cref_t;

    /// @brief Number of unique dimensions in the vectors.
    std::size_t dimensions_ = 0;
    /// @brief Similar to `dimensions_`, but different for fraction byte-length scalars.
    std::size_t scalar_words_ = 0;
    std::size_t expansion_add_ = 0;
    std::size_t expansion_search_ = 0;
    index_t* typed_ = nullptr;

    std::size_t casted_vector_bytes_ = 0;
    mutable std::vector<byte_t> cast_buffer_;
    struct casts_t {
        cast_t from_b1x8;
        cast_t from_f8;
        cast_t from_f16;
        cast_t from_f32;
        cast_t from_f64;

        cast_t to_b1x8;
        cast_t to_f8;
        cast_t to_f16;
        cast_t to_f32;
        cast_t to_f64;
    } casts_;

    metric_t root_metric_;

    mutable std::vector<std::size_t> available_threads_;
    mutable std::mutex available_threads_mutex_;

    using shared_mutex_t = std::mutex; // TODO: Find an OS-compatible solution
    using shared_lock_t = std::unique_lock<shared_mutex_t>;
    using unique_lock_t = std::unique_lock<shared_mutex_t>;

#if USEARCH_LOOKUP_LABEL
    mutable shared_mutex_t labeled_lookup_mutex_;
    tsl::robin_map<label_t, id_t> labeled_lookup_;

    mutable std::mutex free_ids_mutex_;
    ring_gt<id_t> free_ids_;
    label_t free_label_;
#endif

  public:
    using search_result_t = typename index_t::search_result_t;
    using add_result_t = typename index_t::add_result_t;
    using serialization_result_t = typename index_t::serialization_result_t;
    using join_result_t = typename index_t::join_result_t;
    using stats_t = typename index_t::stats_t;
    using match_t = typename index_t::match_t;
    using node_retriever_t = typename index_t::node_retriever_t;

    index_punned_dense_gt() = default;
    index_punned_dense_gt(index_punned_dense_gt&& other)
        : dimensions_(std::move(other.dimensions_)),                   //
          scalar_words_(std::move(other.scalar_words_)),               //
          expansion_add_(std::move(other.expansion_add_)),             //
          expansion_search_(std::move(other.expansion_search_)),       //
          typed_(exchange(other.typed_, nullptr)),                     //
          casted_vector_bytes_(std::move(other.casted_vector_bytes_)), //
          cast_buffer_(std::move(other.cast_buffer_)),                 //
          casts_(std::move(other.casts_)),                             //
          root_metric_(std::move(other.root_metric_)),                 //
          available_threads_(std::move(other.available_threads_))
#if USEARCH_LOOKUP_LABEL
          ,
          labeled_lookup_(std::move(other.labeled_lookup_)), //
          free_ids_(std::move(other.free_ids_)),             //
          free_label_(std::move(other.free_label_))          //
#endif
    {
    } //

    index_punned_dense_gt& operator=(index_punned_dense_gt&& other) {
        swap(other);
        return *this;
    }

    /**
     *  @brief Swaps the contents of this index with another index.
     *  @param other The other index to swap with.
     */
    void swap(index_punned_dense_gt& other) {
        std::swap(dimensions_, other.dimensions_);
        std::swap(scalar_words_, other.scalar_words_);
        std::swap(expansion_add_, other.expansion_add_);
        std::swap(expansion_search_, other.expansion_search_);
        std::swap(typed_, other.typed_);
        std::swap(casted_vector_bytes_, other.casted_vector_bytes_);
        std::swap(cast_buffer_, other.cast_buffer_);
        std::swap(casts_, other.casts_);
        std::swap(root_metric_, other.root_metric_);
        std::swap(available_threads_, other.available_threads_);
#if USEARCH_LOOKUP_LABEL
        std::swap(labeled_lookup_, other.labeled_lookup_);
        std::swap(free_ids_, other.free_ids_);
        std::swap(free_label_, other.free_label_);
#endif
    }

    ~index_punned_dense_gt() {
        if (typed_)
            typed_->~index_t();
        index_allocator_t{}.deallocate(typed_, 1);
        typed_ = nullptr;
    }

    static index_config_t optimize(index_config_t config) { return index_t::optimize(config); }

    /**
     *  @brief Constructs an instance of ::index_punned_dense_gt.
     *  @param[in] dimensions The of dimensions per vector.
     *  @param[in] metric One of the default supported metric @b kinds for distance measurements.
     *  @param[in] config The index configuration (optional).
     *  @param[in] accuracy The scalar kind used for internal representations (optional).
     *  @param[in] expansion_add The expansion factor for adding vectors (optional).
     *  @param[in] expansion_search The expansion factor for searching vectors (optional).
     *  @param[in] free_label The label used for freed vectors (optional).
     *  @return An instance of ::index_punned_dense_gt.
     */
    static index_punned_dense_gt make(                             //
        std::size_t dimensions, metric_kind_t metric,              //
        index_config_t config = {},                                //
        scalar_kind_t accuracy = scalar_kind_t::f32_k,             //
        std::size_t expansion_add = default_expansion_add(),       //
        std::size_t expansion_search = default_expansion_search(), //
        label_t free_label = default_free_value<label_t>()) {

        return make_(                                //
            dimensions, accuracy,                    //
            config, expansion_add, expansion_search, //
            make_metric_(metric, dimensions, accuracy), make_casts_(accuracy), free_label);
    }

    /**
     *  @brief Constructs an instance of ::index_punned_dense_gt.
     *  @param[in] dimensions The of dimensions per vector.
     *  @param[in] metric The @b ad-hoc metric wrapped for third-party distance measures.
     *  @param[in] config The index configuration (optional).
     *  @param[in] accuracy The scalar kind used for internal representations (optional).
     *  @param[in] expansion_add The expansion factor for adding vectors (optional).
     *  @param[in] expansion_search The expansion factor for searching vectors (optional).
     *  @param[in] free_label The label used for freed vectors (optional).
     *  @return An instance of ::index_punned_dense_gt.
     */
    static index_punned_dense_gt make(                             //
        std::size_t dimensions, metric_t metric,                   //
        index_config_t config = {},                                //
        scalar_kind_t accuracy = scalar_kind_t::f32_k,             //
        std::size_t expansion_add = default_expansion_add(),       //
        std::size_t expansion_search = default_expansion_search(), //
        label_t free_label = default_free_value<label_t>()) {

        return make_(                                //
            dimensions, accuracy,                    //
            config, expansion_add, expansion_search, //
            metric_t(metric), make_casts_(accuracy), free_label);
    }

    std::size_t dimensions() const { return dimensions_; }
    std::size_t scalar_words() const { return scalar_words_; }
    std::size_t connectivity() const { return typed_->connectivity(); }
    std::size_t size() const {
#if USEARCH_LOOKUP_LABEL
        return typed_->size() - free_ids_.size();
#else
        return typed_->size();
#endif
    }
    std::size_t capacity() const { return typed_->capacity(); }
    std::size_t max_level() const noexcept { return typed_->max_level(); }
    index_config_t const& config() const { return typed_->config(); }
    index_limits_t const& limits() const { return typed_->limits(); }

    metric_t const& metric() const { return root_metric_; }
    std::size_t expansion_add() const { return expansion_add_; }
    std::size_t expansion_search() const { return expansion_search_; }
    void change_expansion_add(std::size_t n) { expansion_add_ = n; }
    void change_expansion_search(std::size_t n) { expansion_search_ = n; }

    member_citerator_t cbegin() const { return typed_->cbegin(); }
    member_citerator_t cend() const { return typed_->cend(); }
    member_citerator_t begin() const { return typed_->begin(); }
    member_citerator_t end() const { return typed_->end(); }
    member_iterator_t begin() { return typed_->begin(); }
    member_iterator_t end() { return typed_->end(); }

    stats_t stats() const { return typed_->stats(); }
    stats_t stats(std::size_t level) const { return typed_->stats(level); }
    precomputed_constants_t metadata() const { return typed_->metadata(); }

    std::size_t memory_usage() const {
        return typed_->memory_usage(0) +                 //
               typed_->tape_allocator().total_wasted() + //
               typed_->tape_allocator().total_reserved();
    }

    // clang-format off
    add_result_t add(label_t label, b1x8_t const* vector) { return add_(label, vector, casts_.from_b1x8); }
    add_result_t add(label_t label, f8_bits_t const* vector) { return add_(label, vector, casts_.from_f8); }
    add_result_t add(label_t label, f16_t const* vector) { return add_(label, vector, casts_.from_f16); }
    add_result_t add(label_t label, f32_t const* vector, int32_t level = -1, byte_t *tape = nullptr) { return add_(label, vector, casts_.from_f32, level, tape); }
    add_result_t add(label_t label, f64_t const* vector) { return add_(label, vector, casts_.from_f64); }

    add_result_t add(label_t label, b1x8_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_b1x8); }
    add_result_t add(label_t label, f8_bits_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_f8); }
    add_result_t add(label_t label, f16_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_f16); }
    add_result_t add(label_t label, f32_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_f32); }
    add_result_t add(label_t label, f64_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_f64); }

    search_result_t search(b1x8_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_b1x8); }
    search_result_t search(f8_bits_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f8); }
    search_result_t search(f16_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f16); }
    search_result_t search(f32_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f32); }
    search_result_t search(f64_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f64); }

    search_result_t search(b1x8_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_b1x8); }
    search_result_t search(f8_bits_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_f8); }
    search_result_t search(f16_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_f16); }
    search_result_t search(f32_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_f32); }
    search_result_t search(f64_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_f64); }

#if USEARCH_LOOKUP_LABEL
    bool get(label_t label, b1x8_t* vector) const { return get_(label, vector, casts_.to_b1x8); }
    bool get(label_t label, f8_bits_t* vector) const { return get_(label, vector, casts_.to_f8); }
    bool get(label_t label, f16_t* vector) const { return get_(label, vector, casts_.to_f16); }
    bool get(label_t label, f32_t* vector) const { return get_(label, vector, casts_.to_f32); }
    bool get(label_t label, f64_t* vector) const { return get_(label, vector, casts_.to_f64); }
#endif
    // clang-format on

    search_result_t empty_search_result() const { return search_result_t{*typed_}; }

    /**
     *  @brief Reserves memory for the index and the labeled lookup.
     *  @return `true` if the memory reservation was successful, `false` otherwise.
     */
    bool reserve(index_limits_t limits) {
#if USEARCH_LOOKUP_LABEL
        {
            unique_lock_t lock(labeled_lookup_mutex_);
            labeled_lookup_.reserve(limits.members);
        }
#endif
        return typed_->reserve(limits);
    }

    /**
     *  @brief Clears the whole index, reclaiming the memory.
     */
    void clear() {
#if USEARCH_LOOKUP_LABEL
        unique_lock_t lookup_lock(labeled_lookup_mutex_);
        std::unique_lock<std::mutex> free_lock(free_ids_mutex_);
        labeled_lookup_.clear();
        free_ids_.clear();
#endif
        typed_->clear();
    }

    /**
     *  @brief Saves the index to a file.
     *  @param[in] path The path to the file.
     *  @return Outcome descriptor explictly convertable to boolean.
     */
    serialization_result_t save(char const* path, char** usearch_result_buf) const {
        return typed_->save(path, usearch_result_buf, this->expansion_search(), this->expansion_add());
    }

    /**
     *  @brief Parses the index from file to RAM.
     *  @param[in] path The path to the file.
     *  @return Outcome descriptor explictly convertable to boolean.
     */
    serialization_result_t load(char const* path) {
        serialization_result_t result = typed_->load(path);
        index_config_t config = typed_->config();
        metric_t metric = make_metric_(config.metric_kind, config.dimensions, config.accuracy);
        typed_->change_metric(metric);
        root_metric_ = metric;
        this->change_expansion_add(config.expansion_add);
        this->change_expansion_search(config.expansion_search);
#if USEARCH_LOOKUP_LABEL
        if (result)
            reindex_labels_();
#endif
        return result;
    }

    /**
     *  @brief Parses the index from file, without loading it into RAM.
     *  @param[in] path The path to the file.
     *  @return Outcome descriptor explictly convertable to boolean.
     */
    serialization_result_t view(char const* path) {
        serialization_result_t result = typed_->view(path);
#if USEARCH_LOOKUP_LABEL
        if (result)
            reindex_labels_();
#endif
        return result;
    }

    serialization_result_t view_mem(char* memory) {
        serialization_result_t result = typed_->view_mem(memory);
#if USEARCH_LOOKUP_LABEL
        if (result)
            reindex_labels_();
#endif
        return result;
    }

    serialization_result_t view_mem_lazy(char* memory) {
        serialization_result_t result;
#if USEARCH_LOOKUP_LABEL
        return result.failed("Usearch does not support label lookup and member removals for external memory indexes.");
#endif
        result = typed_->view_mem_lazy(memory);
        return std::move(result);
    }

    serialization_result_t update_header(char* headerp) {
        serialization_result_t result = typed_->update_header(headerp);
        return result;
    }

    void set_node_retriever(void* retriever_ctx, node_retriever_t node_retriever, node_retriever_t node_retriever_mut) {
        typed_->set_node_retriever(retriever_ctx, node_retriever, node_retriever_mut);
    }

#if USEARCH_LOOKUP_LABEL
    /**
     *  @brief Checks if a vector with specidied label is present.
     *  @return `true` if the label is present in the index, `false` otherwise.
     */
    bool contains(label_t label) const {
        shared_lock_t lock(labeled_lookup_mutex_);
        return labeled_lookup_.contains(label);
    }
#endif

    struct labeling_result_t {
        error_t error{};
        std::size_t completed{};

        explicit operator bool() const noexcept { return !error; }
        labeling_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

#if USEARCH_LOOKUP_LABEL
    /**
     *  @brief Removes an entry with the specified label from the index.
     *  @param[in] label The label of the entry to remove.
     *  @return The ::labeling_result_t indicating the result of the removal operation.
     *          If the removal was successful, `result.completed` will be `true`.
     *          If the label was not found in the index, `result.completed` will be `false`.
     *          If an error occurred during the removal operation, `result.error` will contain an error message.
     */
    labeling_result_t remove(label_t label) {
        labeling_result_t result;

        unique_lock_t lookup_lock(labeled_lookup_mutex_);
        auto id_terator = labeled_lookup_.find(label);
        if (id_terator == labeled_lookup_.end())
            return result;

        // Grow the removed entries ring, if needed
        std::unique_lock<std::mutex> free_lock(free_ids_mutex_);
        if (free_ids_.size() == free_ids_.capacity())
            if (!free_ids_.reserve((std::max<std::size_t>)(free_ids_.capacity() * 2, 64ul)))
                return result.failed("Can't allocate memory for a free-list");

        // A removed entry would be:
        // - present in `free_ids_`
        // - missing in the `labeled_lookup_`
        // - marked in the `typed_` index with a `free_label_`
        id_t id = id_terator->second;
        free_ids_.push(id);
        labeled_lookup_.erase(id_terator);
        typed_->at(id).label = free_label_;
        result.completed = true;

        return result;
    }

    /**
     *  @brief Removes multiple entries with the specified labels from the index.
     *  @param[in] labels_begin The beginning of the labels range.
     *  @param[in] labels_end The ending of the labels range.
     *  @return The ::labeling_result_t indicating the result of the removal operation.
     *          `result.completed` will contain the number of labels that were successfully removed.
     *          `result.error` will contain an error message if an error occurred during the removal operation.
     */
    template <typename labels_iterator_at>
    labeling_result_t remove(labels_iterator_at&& labels_begin, labels_iterator_at&& labels_end) {

        labeling_result_t result;
        unique_lock_t lookup_lock(labeled_lookup_mutex_);
        std::unique_lock<std::mutex> free_lock(free_ids_mutex_);

        // Grow the removed entries ring, if needed
        std::size_t count_requests = std::distance(labels_begin, labels_end);
        if (!free_ids_.reserve(free_ids_.size() + count_requests))
            return result.failed("Can't allocate memory for a free-list");

        // Remove them one-by-one
        for (auto label_it = labels_begin; label_it != labels_end; ++label_it) {
            label_t label = *label_it;
            auto id_terator = labeled_lookup_.find(label);
            if (id_terator == labeled_lookup_.end())
                continue;

            // A removed entry would be:
            // - present in `free_ids_`
            // - missing in the `labeled_lookup_`
            // - marked in the `typed_` index with a `free_label_`
            id_t id = id_terator->second;
            free_ids_.push(id);
            labeled_lookup_.erase(id_terator);
            typed_->at(id).label = free_label_;
            result.completed += 1;
        }

        return result;
    }

    /**
     *  @brief Renames an entry with the specified label to a new label.
     *  @param[in] from The current label of the entry to rename.
     *  @param[in] to The new label to assign to the entry.
     *  @return The ::labeling_result_t indicating the result of the rename operation.
     *          If the rename was successful, `result.completed` will be `true`.
     *          If the entry with the current label was not found, `result.completed` will be `false`.
     */
    labeling_result_t rename(label_t from, label_t to) {
        labeling_result_t result;
        unique_lock_t lookup_lock(labeled_lookup_mutex_);
        auto id_terator = labeled_lookup_.find(from);
        if (id_terator == labeled_lookup_.end())
            return result;

        id_t id = id_terator->second;
        labeled_lookup_.erase(id_terator);
        labeled_lookup_.emplace(to, id);
        typed_->at(id).label = to;
        result.completed = true;
        return result;
    }

    /**
     *  @brief Exports a range of labels for the vectors present in the index.
     *  @param[out] labels Pointer to the array where the labels will be exported.
     *  @param[in] offset The number of labels to skip. Useful for pagination.
     *  @param[in] limit The maximum number of labels to export, that can fit in ::labels.
     */
    void export_labels(label_t* labels, std::size_t offset, std::size_t limit) const {
        shared_lock_t lock(labeled_lookup_mutex_);
        auto it = labeled_lookup_.begin();
        offset = (std::min)(offset, labeled_lookup_.size());
        std::advance(it, offset);
        for (; it != labeled_lookup_.end() && limit; ++it, ++labels, --limit)
            *labels = it->first;
    }
#endif
    int32_t newnode_level() {
        thread_lock_t lock = thread_lock_();
        add_config_t add_config;
        add_config.thread = lock.thread_id;
        return typed_->choose_random_level(add_config);
    }

    /**
     *  @brief  Adapts the Male-Optimal Stable Marriage algorithm for unequal sets
     *          to perform fast one-to-one matching between two large collections
     *          of vectors, using approximate nearest neighbors search.
     *
     *  @param[inout] first_to_second Container to map ::first labels to ::second.
     *  @param[inout] second_to_first Container to map ::second labels to ::first.
     *  @param[in] executor Thread-pool to execute the job in parallel.
     *  @param[in] progress Callback to report the execution progress.
     */
    template <                                                        //
        typename first_to_second_at = dummy_label_to_label_mapping_t, //
        typename second_to_first_at = dummy_label_to_label_mapping_t, //
        typename executor_at = dummy_executor_t,                      //
        typename progress_at = dummy_progress_t                       //
        >
    static join_result_t join(                                       //
        index_punned_dense_gt const& first,                          //
        index_punned_dense_gt const& second,                         //
        join_config_t config = {},                                   //
        first_to_second_at&& first_to_second = first_to_second_at{}, //
        second_to_first_at&& second_to_first = second_to_first_at{}, //
        executor_at&& executor = executor_at{},                      //
        progress_at&& progress = progress_at{}) noexcept {

        return index_t::join(                                  //
            *first.typed_, *second.typed_, config,             //
            std::forward<first_to_second_at>(first_to_second), //
            std::forward<second_to_first_at>(second_to_first), //
            std::forward<executor_at>(executor),               //
            std::forward<progress_at>(progress));
    }

    struct copy_result_t {
        index_punned_dense_gt index;
        error_t error;

        explicit operator bool() const noexcept { return !error; }
        copy_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

    /**
     *  @brief Copies the ::index_punned_dense_gt @b with all the data in it.
     *  @param config The copy configuration (optional).
     *  @return A copy of the ::index_punned_dense_gt instance.
     */
    copy_result_t copy(copy_config_t config = {}) const {
        copy_result_t result = fork();
        if (!result)
            return result;
        auto typed_result = typed_->copy(config);
        if (!typed_result)
            return result.failed(std::move(typed_result.error));
#if USEARCH_LOOKUP_LABEL
        if (!result.index.free_ids_.reserve(free_ids_.size()))
            return result.failed(std::move(typed_result.error));
        for (std::size_t i = 0; i != free_ids_.size(); ++i)
            result.index.free_ids_.push(free_ids_[i]);

        // q:: where would copy be used? Can the old index be used while/after
        // copying? if so, should we not keep track of the old lock as well?
        result.index.labeled_lookup_ = labeled_lookup_;
#endif
        *result.index.typed_ = std::move(typed_result.index);
        return result;
    }

    /**
     *  @brief Copies the ::index_punned_dense_gt model @b without any data.
     *  @return A similarly configured ::index_punned_dense_gt instance.
     */
    copy_result_t fork() const {
        copy_result_t result;
        index_punned_dense_gt& other = result.index;

        other.dimensions_ = dimensions_;
        other.scalar_words_ = scalar_words_;
        other.expansion_add_ = expansion_add_;
        other.expansion_search_ = expansion_search_;
        other.casted_vector_bytes_ = casted_vector_bytes_;
        other.cast_buffer_ = cast_buffer_;
        other.casts_ = casts_;

        other.root_metric_ = root_metric_;
        other.available_threads_ = available_threads_;

        index_t* raw = index_allocator_t{}.allocate(1);
        if (!raw)
            return result.failed("Can't allocate the index");

        new (raw) index_t(config(), root_metric_);
        other.typed_ = raw;
        return result;
    }

    struct compaction_result_t {
        error_t error{};
        std::size_t pruned_edges{};

        explicit operator bool() const noexcept { return !error; }
        compaction_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

#if USEARCH_LOOKUP_LABEL
    /**
     *  @brief Performs compaction on the index, pruning links to removed entries.
     *  @param executor The executor parallel processing. Default ::dummy_executor_t single-threaded.
     *  @param progress The progress tracker instance to use. Default ::dummy_progress_t reports nothing.
     *  @return The ::compaction_result_t indicating the result of the compaction operation.
     *          `result.pruned_edges` will contain the number of edges that were removed.
     *          `result.error` will contain an error message if an error occurred during the compaction operation.
     */
    template <typename executor_at = dummy_executor_t, typename progress_at = dummy_progress_t>
    compaction_result_t compact(executor_at&& executor = executor_at{}, progress_at&& progress = progress_at{}) {
        compaction_result_t result;
        std::atomic<std::size_t> pruned_edges;
        auto disallow = [&](member_cref_t const& member) noexcept {
            bool freed = member.label == free_label_;
            pruned_edges += freed;
            return freed;
        };
        typed_->isolate(disallow, std::forward<executor_at>(executor), std::forward<progress_at>(progress));
        result.pruned_edges = pruned_edges;
        return result;
    }
#endif

    static metric_t make_metric_(metric_kind_t kind, std::size_t dimensions, scalar_kind_t accuracy) {
        switch (kind) {
        case metric_kind_t::ip_k: return ip_metric_(dimensions, accuracy);
        case metric_kind_t::cos_k: return cos_metric_(dimensions, accuracy);
        case metric_kind_t::l2sq_k: return l2sq_metric_(dimensions, accuracy);
        case metric_kind_t::pearson_k: return pearson_metric_(accuracy);
        case metric_kind_t::haversine_k: return haversine_metric_(accuracy);
        case metric_kind_t::hamming_k: return hamming_gt<b1x8_t>{};
        case metric_kind_t::jaccard_k: // Equivalent to Tanimoto
        case metric_kind_t::tanimoto_k: return tanimoto_gt<b1x8_t>{};
        case metric_kind_t::sorensen_k: return sorensen_gt<b1x8_t>{};
        default: return {};
        }
    }

  private:
    struct thread_lock_t {
        index_punned_dense_gt const& parent;
        std::size_t thread_id;

        ~thread_lock_t() { parent.thread_unlock_(thread_id); }
    };

    thread_lock_t thread_lock_() const {
        available_threads_mutex_.lock();
        std::size_t thread_id = available_threads_.back();
        available_threads_.pop_back();
        available_threads_mutex_.unlock();
        return {*this, thread_id};
    }

    void thread_unlock_(std::size_t thread_id) const {
        available_threads_mutex_.lock();
        available_threads_.push_back(thread_id);
        available_threads_mutex_.unlock();
    }

    template <typename scalar_at>
    add_result_t add_(label_t label, scalar_at const* vector, add_config_t config, cast_t const& cast,
                      int32_t level = -1, byte_t* tape = nullptr) {
        byte_t const* vector_data = reinterpret_cast<byte_t const*>(vector);
        std::size_t vector_bytes = dimensions_ * sizeof(scalar_at);

        byte_t* casted_data = cast_buffer_.data() + casted_vector_bytes_ * config.thread;
        bool casted = cast(vector_data, dimensions_, casted_data);
        if (casted)
            vector_data = casted_data, vector_bytes = casted_vector_bytes_, config.store_vector = true;

        // Check if there are some removed entries, whose nodes we can reuse
        id_t free_id = default_free_value<id_t>();
#if USEARCH_LOOKUP_LABEL
        {
            std::unique_lock<std::mutex> lock(free_ids_mutex_);
            free_ids_.try_pop(free_id);
        }
#endif

        // Perform the insertion or the update
        add_result_t result =                     //
            free_id != default_free_value<id_t>() //
                ? typed_->update(free_id, label, {vector_data, vector_bytes}, config)
                : typed_->add(label, {vector_data, vector_bytes}, config, level, tape);
#if USEARCH_LOOKUP_LABEL
        {
            unique_lock_t lock(labeled_lookup_mutex_);
            labeled_lookup_.emplace(label, result.id);
        }
#endif
        return result;
    }

    template <typename scalar_at>
    search_result_t search_(                         //
        scalar_at const* vector, std::size_t wanted, //
        search_config_t config, cast_t const& cast) const {

        byte_t const* vector_data = reinterpret_cast<byte_t const*>(vector);
        std::size_t vector_bytes = dimensions_ * sizeof(scalar_at);

        byte_t* casted_data = cast_buffer_.data() + casted_vector_bytes_ * config.thread;
        bool casted = cast(vector_data, dimensions_, casted_data);
        if (casted)
            vector_data = casted_data, vector_bytes = casted_vector_bytes_;

#if USEARCH_LOOKUP_LABEL
        auto allow = [=](match_t const& match) noexcept { return match.member.label != free_label_; };
        return typed_->search({vector_data, vector_bytes}, wanted, config, allow);
#else
        return typed_->search({vector_data, vector_bytes}, wanted, config);
#endif
    }

#if USEARCH_LOOKUP_LABEL
    id_t lookup_id_(label_t label) const {
        shared_lock_t lock(labeled_lookup_mutex_);
        return labeled_lookup_.at(label);
    }
#endif

#if USEARCH_LOOKUP_LABEL
    void reindex_labels_() {

        // Estimate number of entries first
        std::size_t count_total = typed_->size();
        std::size_t count_removed = 0;
        for (std::size_t i = 0; i != count_total; ++i) {
            member_cref_t member = typed_->at(i);
            count_removed += member.label == free_label_;
        }

        unique_lock_t lock(labeled_lookup_mutex_);
        labeled_lookup_.clear();
        labeled_lookup_.reserve(count_total - count_removed);
        free_ids_.clear();
        free_ids_.reserve(count_removed);
        for (std::size_t i = 0; i != typed_->size(); ++i) {
            member_cref_t member = typed_->at(i);
            if (member.label == free_label_)
                free_ids_.push(static_cast<id_t>(i));
            else
                labeled_lookup_.emplace(member.label, static_cast<id_t>(i));
        }
    }

    template <typename scalar_at> bool get_(label_t label, scalar_at* reconstructed, cast_t const& cast) const {
        id_t id;
        // Find the matching ID
        {
            shared_lock_t lock(labeled_lookup_mutex_);
            auto it = labeled_lookup_.find(label);
            if (it == labeled_lookup_.end())
                return false;
            id = it->second;
        }
        // Export the entry
        member_cref_t member = typed_->at(id);
        byte_t const* punned_vector = reinterpret_cast<byte_t const*>(member.vector.data());
        bool casted = cast(punned_vector, dimensions_, (byte_t*)reconstructed);
        if (!casted)
            std::memcpy(reconstructed, punned_vector, casted_vector_bytes_);
        return true;
    }
#endif

    template <typename scalar_at>
    add_result_t add_(label_t label, scalar_at const* vector, cast_t const& cast, int32_t level = -1,
                      byte_t* tape = nullptr) {

        std::size_t thread_id = 0;
#if USEARCH_CONCURRENT
        thread_lock_t lock = thread_lock_();
        thread_id = lock.thread_id;
#endif
        add_config_t add_config;
        add_config.expansion = expansion_add_;
        add_config.thread = thread_id;
        return add_(label, vector, add_config, cast, level, tape);
    }

    template <typename scalar_at>
    search_result_t search_(                         //
        scalar_at const* vector, std::size_t wanted, //
        cast_t const& cast) const {
        thread_lock_t lock = thread_lock_();
        search_config_t search_config;
        search_config.expansion = expansion_search_;
        search_config.thread = lock.thread_id;
        return search_(vector, wanted, search_config, cast);
    }

    static index_punned_dense_gt make_(                                                 //
        std::size_t dimensions, scalar_kind_t scalar_kind,                              //
        index_config_t config, std::size_t expansion_add, std::size_t expansion_search, //
        metric_t metric, casts_t casts, label_t free_label) {

#if USEARCH_CONCURRENT
        std::size_t hardware_threads = std::thread::hardware_concurrency();
#else
        std::size_t hardware_threads = 1;
#endif
        index_punned_dense_gt result;
        result.dimensions_ = dimensions;
        result.scalar_words_ = count_scalar_words_(dimensions, scalar_kind);
        result.expansion_add_ = expansion_add;
        result.expansion_search_ = expansion_search;
        result.casted_vector_bytes_ = bytes_per_scalar(scalar_kind) * result.scalar_words_;
        result.cast_buffer_.resize(hardware_threads * result.casted_vector_bytes_);
        result.casts_ = casts;
        result.root_metric_ = metric;
#if USEARCH_LOOKUP_LABEL
        result.free_label_ = free_label;
#endif

        // Fill the thread IDs.
        result.available_threads_.resize(hardware_threads);
        std::iota(result.available_threads_.begin(), result.available_threads_.end(), 0ul);

        // Available since C11, but only C++17, so we use the C version.
        index_t* raw = index_allocator_t{}.allocate(1);
        new (raw) index_t(config, metric);
        result.typed_ = raw;
        return result;
    }

    template <typename to_scalar_at> static casts_t make_casts_() {
        casts_t result;

        result.from_b1x8 = cast_gt<b1x8_t, to_scalar_at>{};
        result.from_f8 = cast_gt<f8_bits_t, to_scalar_at>{};
        result.from_f16 = cast_gt<f16_t, to_scalar_at>{};
        result.from_f32 = cast_gt<f32_t, to_scalar_at>{};
        result.from_f64 = cast_gt<f64_t, to_scalar_at>{};

        result.to_b1x8 = cast_gt<to_scalar_at, b1x8_t>{};
        result.to_f8 = cast_gt<to_scalar_at, f8_bits_t>{};
        result.to_f16 = cast_gt<to_scalar_at, f16_t>{};
        result.to_f32 = cast_gt<to_scalar_at, f32_t>{};
        result.to_f64 = cast_gt<to_scalar_at, f64_t>{};

        return result;
    }

    static casts_t make_casts_(scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f64_k: return make_casts_<f64_t>();
        case scalar_kind_t::f32_k: return make_casts_<f32_t>();
        case scalar_kind_t::f16_k: return make_casts_<f16_t>();
        case scalar_kind_t::f8_k: return make_casts_<f8_bits_t>();
        case scalar_kind_t::b1x8_k: return make_casts_<b1x8_t>();
        default: return {};
        }
    }

    static std::size_t count_scalar_words_(std::size_t dimensions, scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f64_k: return dimensions;
        case scalar_kind_t::f32_k: return dimensions;
        case scalar_kind_t::f16_k: return dimensions;
        case scalar_kind_t::f8_k: return dimensions;
        case scalar_kind_t::b1x8_k: return divide_round_up<CHAR_BIT>(dimensions);
        default: return {};
        }
    }

    static metric_t ip_metric_f32_(std::size_t dimensions) {
        (void)dimensions;
#if USEARCH_USE_SIMSIMD
        if (hardware_supports(isa_t::avx2_k) && dimensions % 4 == 0)
            return {metric_kind_t::ip_k, isa_t::avx2_k, 0.f,
                    [=](f32_t const* a, f32_t const* b) { return 1.f - simsimd_dot_f32x4avx2(a, b, dimensions); }};
        if (hardware_supports(isa_t::sve_k))
            return {metric_kind_t::ip_k, isa_t::sve_k, 0.f,
                    [=](f32_t const* a, f32_t const* b) { return 1.f - simsimd_dot_f32sve(a, b, dimensions); }};
        if (hardware_supports(isa_t::neon_k) && dimensions % 4 == 0)
            return {metric_kind_t::ip_k, isa_t::neon_k, 0.f,
                    [=](f32_t const* a, f32_t const* b) { return 1.f - simsimd_dot_f32x4neon(a, b, dimensions); }};
#endif
        return ip_gt<f32_t>{};
    }

    static metric_t cos_metric_f16_(std::size_t dimensions) {
        (void)dimensions;
#if USEARCH_USE_SIMSIMD
        if (hardware_supports(isa_t::avx512_k) && dimensions % 16 == 0)
            return {metric_kind_t::cos_k, isa_t::avx512_k, simsimd_f16_t(0),
                    [=](simsimd_f16_t const* a, simsimd_f16_t const* b) {
                        return 1.f - simsimd_cos_f16x16avx512(a, b, dimensions);
                    }};
        if (hardware_supports(isa_t::neon_k) && dimensions % 4 == 0)
            return {metric_kind_t::cos_k, isa_t::neon_k, simsimd_f16_t(0),
                    [=](simsimd_f16_t const* a, simsimd_f16_t const* b) {
                        return 1.f - simsimd_cos_f16x4neon(a, b, dimensions);
                    }};
#endif
        return cos_gt<f16_t, f32_t>{};
    }

    static metric_t cos_metric_f8_(std::size_t dimensions) {
#if USEARCH_USE_SIMSIMD
        if (hardware_supports(isa_t::neon_k) && dimensions % 16 == 0)
            return {metric_kind_t::cos_k, isa_t::neon_k, int8_t(0),
                    [=](int8_t const* a, int8_t const* b) { return 1.f - simsimd_cos_i8x16neon(a, b, dimensions); }};
#endif
        return cos_f8_t{dimensions};
    }

    static metric_t ip_metric_(std::size_t dimensions, scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f32_k:
            // The two most common numeric types for the most common metric have optimized versions
            return ip_metric_f32_(dimensions);
        case scalar_kind_t::f16_k:
            // Dot-product accumulates error, Cosine-distance normalizes it
            return cos_metric_f16_(dimensions);

        case scalar_kind_t::f8_k: return cos_metric_f8_(dimensions);
        case scalar_kind_t::f64_k: return ip_gt<f64_t>{};
        default: return {};
        }
    }

    static metric_t l2sq_metric_(std::size_t dimensions, scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f8_k: return l2sq_f8_t{dimensions};
        case scalar_kind_t::f16_k: return l2sq_gt<f16_t, f32_t>{};
        case scalar_kind_t::f32_k: return l2sq_gt<f32_t>{};
        case scalar_kind_t::f64_k: return l2sq_gt<f64_t>{};
        default: return {};
        }
    }

    static metric_t cos_metric_(std::size_t dimensions, scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f8_k: return cos_metric_f8_(dimensions);
        case scalar_kind_t::f16_k: return cos_metric_f16_(dimensions);
        case scalar_kind_t::f32_k: return cos_gt<f32_t>{};
        case scalar_kind_t::f64_k: return cos_gt<f64_t>{};
        default: return {};
        }
    }

    static metric_t haversine_metric_(scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f8_k: return haversine_gt<f8_bits_t, f32_t>{};
        case scalar_kind_t::f16_k: return haversine_gt<f16_t, f32_t>{};
        case scalar_kind_t::f32_k: return haversine_gt<f32_t>{};
        case scalar_kind_t::f64_k: return haversine_gt<f64_t>{};
        default: return {};
        }
    }

    static metric_t pearson_metric_(scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f8_k: return pearson_correlation_gt<f8_bits_t, f32_t>{};
        case scalar_kind_t::f16_k: return pearson_correlation_gt<f16_t, f32_t>{};
        case scalar_kind_t::f32_k: return pearson_correlation_gt<f32_t>{};
        case scalar_kind_t::f64_k: return pearson_correlation_gt<f64_t>{};
        default: return {};
        }
    }
};

using punned_small_t = index_punned_dense_gt<std::uint64_t, std::uint32_t>;
using punned_big_t = index_punned_dense_gt<uuid_t, uint40_t>;

} // namespace usearch
} // namespace unum
