#include "lib.hpp"
#include "usearch/rust/lib.rs.h"

using namespace unum::usearch;
using namespace unum;

using index_t = typename Index::index_t;
using add_result_t = typename index_t::add_result_t;
using search_result_t = typename index_t::search_result_t;
using serialization_result_t = typename index_t::serialization_result_t;

Index::Index(std::unique_ptr<index_t> index) : index_(std::move(index)) {}

void Index::add_in_thread(label_t label, rust::Slice<float const> vector, size_t thread) const {
    add_config_t config;
    config.thread = thread;
    index_->add(label, vector.data(), config).error.raise();
}

Matches Index::search_in_thread(rust::Slice<float const> vector, size_t count, size_t thread) const {
    Matches matches;
    matches.labels.reserve(count);
    matches.distances.reserve(count);
    for (size_t i = 0; i != count; ++i)
        matches.labels.push_back(0), matches.distances.push_back(0);
    search_config_t config;
    config.thread = thread;
    search_result_t result = index_->search(vector.data(), count, config);
    result.error.raise();
    matches.count = result.dump_to(matches.labels.data(), matches.distances.data());
    matches.labels.truncate(matches.count);
    matches.distances.truncate(matches.count);
    return matches;
}

void Index::add(label_t label, rust::Slice<float const> vector) const {
    index_->add(label, vector.data()).error.raise();
}

Matches Index::search(rust::Slice<float const> vector, size_t count) const {
    Matches matches;
    matches.labels.reserve(count);
    matches.distances.reserve(count);
    for (size_t i = 0; i != count; ++i)
        matches.labels.push_back(0), matches.distances.push_back(0);
    search_result_t result = index_->search(vector.data(), count);
    result.error.raise();
    matches.count = result.dump_to(matches.labels.data(), matches.distances.data());
    matches.labels.truncate(matches.count);
    matches.distances.truncate(matches.count);
    return matches;
}

void Index::reserve(size_t capacity) const { index_->reserve(capacity); }

size_t Index::dimensions() const { return index_->dimensions(); }
size_t Index::connectivity() const { return index_->connectivity(); }
size_t Index::size() const { return index_->size(); }
size_t Index::capacity() const { return index_->capacity(); }

void Index::save(rust::Str path) const { index_->save(std::string(path).c_str(), NULL); }
void Index::load(rust::Str path) const { index_->load(std::string(path).c_str()); }
void Index::view(rust::Str path) const { index_->view(std::string(path).c_str()); }

scalar_kind_t accuracy(rust::Str quant) { return scalar_kind_from_name(quant.data(), quant.size()); }

std::unique_ptr<Index> wrap(index_t&& index) {
    std::unique_ptr<index_t> punned_ptr;
    punned_ptr.reset(new index_t(std::move(index)));
    std::unique_ptr<Index> result;
    result.reset(new Index(std::move(punned_ptr)));
    return result;
}

index_config_t config(size_t connectivity) {
    index_config_t result;
    result.connectivity = connectivity ? connectivity : default_connectivity();
    return result;
}

metric_kind_t rust_to_cpp_metric(MetricKind value) {
    switch (value) {
    case MetricKind::IP: return metric_kind_t::ip_k;
    case MetricKind::L2Sq: return metric_kind_t::l2sq_k;
    case MetricKind::Cos: return metric_kind_t::cos_k;
    case MetricKind::Pearson: return metric_kind_t::pearson_k;
    case MetricKind::Haversine: return metric_kind_t::haversine_k;
    case MetricKind::Hamming: return metric_kind_t::hamming_k;
    case MetricKind::Tanimoto: return metric_kind_t::tanimoto_k;
    case MetricKind::Sorensen: return metric_kind_t::sorensen_k;
    default: return metric_kind_t::unknown_k;
    }
}

scalar_kind_t rust_to_cpp_scalar(ScalarKind value) {
    switch (value) {
    case ScalarKind::F8: return scalar_kind_t::f8_k;
    case ScalarKind::F16: return scalar_kind_t::f16_k;
    case ScalarKind::F32: return scalar_kind_t::f32_k;
    case ScalarKind::F64: return scalar_kind_t::f64_k;
    case ScalarKind::B1: return scalar_kind_t::b1x8_k;
    default: return scalar_kind_t::unknown_k;
    }
}

std::unique_ptr<Index> new_index(IndexOptions const& options) {
    return wrap(index_t::make(options.dimensions, rust_to_cpp_metric(options.metric), config(options.connectivity),
                              rust_to_cpp_scalar(options.quantization), options.expansion_add,
                              options.expansion_search));
}
