// Adapter forwarding sonder::backends::llamacpp PQ2_0 -> Q2_0 APIs to the
// core sonder::inference implementation (include/sonder/inference/pq2_reblock.hpp)
// while cross-checking Q2_0 dequantization against upstream ggml's type traits.
#include "pq2_reblock.hpp"

#include <cstring>

#include "ggml.h"
#include "sonder/inference/pq2_reblock.hpp"

namespace sonder::backends::llamacpp {

namespace {

namespace si = sonder::inference;

[[nodiscard]] Status MapStatus(const si::Status& st) {
    if (st.ok()) return Status::Ok();
    switch (st.code()) {
        case si::ErrorCode::not_found:
            return Status::Error(ErrorCode::kFileNotFound, st.message());
        case si::ErrorCode::unsupported:
            return Status::Error(ErrorCode::kUnsupported, st.message());
        case si::ErrorCode::invalid_argument:
        case si::ErrorCode::io_error:
        default:
            return Status::Error(ErrorCode::kInvalidArgument, st.message());
    }
}

[[nodiscard]] Pq2GgufInspection MapInspection(const si::Pq2GgufInspection& s) {
    Pq2GgufInspection d;
    d.gguf_version = s.gguf_version;
    d.tensor_count = s.tensor_count;
    d.pq2_tensor_count = s.pq2_tensor_count;
    d.q2_tensor_count = s.q2_tensor_count;
    d.pq2_input_bytes = s.pq2_input_bytes;
    d.q2_output_bytes = s.q2_output_bytes;
    d.other_tensor_bytes = s.other_tensor_bytes;
    d.prism_hadamard = s.prism_hadamard;
    d.file_type = s.file_type;
    return d;
}

}  // namespace

Status DequantizePq2Row(std::span<const std::uint8_t> pq2_bytes,
                        std::size_t n_elements,
                        std::vector<float>& out_floats) {
    return MapStatus(si::dequantize_pq2_row(pq2_bytes, n_elements, out_floats));
}

Status ReblockPq2RowToQ2(std::span<const std::uint8_t> pq2_bytes,
                         std::size_t n_elements,
                         std::vector<std::uint8_t>& out_q2_bytes) {
    return MapStatus(si::reblock_pq2_row_to_q2(pq2_bytes, n_elements, out_q2_bytes));
}

Status InspectPq2Gguf(const std::filesystem::path& path,
                      Pq2GgufInspection& out_inspection) {
    si::Pq2GgufInspection core_insp;
    const si::Status st = si::inspect_pq2_gguf(path, core_insp);
    if (!st.ok()) {
        out_inspection = {};
        return MapStatus(st);
    }
    out_inspection = MapInspection(core_insp);
    return Status::Ok();
}

Status ReblockPq2ToQ2Gguf(const std::filesystem::path& input_path,
                          const std::filesystem::path& output_path,
                          Pq2ReblockReport* out_report) {
    si::Pq2ReblockReport core_rep;
    const si::Status st = si::reblock_pq2_to_q2_gguf(input_path, output_path, &core_rep);
    if (!st.ok()) {
        if (out_report != nullptr) *out_report = {};
        return MapStatus(st);
    }
    if (out_report != nullptr) {
        out_report->inspection = MapInspection(core_rep.inspection);
        out_report->reblocked_tensors = core_rep.reblocked_tensors;
        out_report->copied_tensors = core_rep.copied_tensors;
        out_report->output_file_bytes = core_rep.output_file_bytes;
        out_report->output_sha256 = core_rep.output_sha256;
    }
    return Status::Ok();
}

Status VerifyPq2ToQ2Gguf(const std::filesystem::path& pq2_path,
                         const std::filesystem::path& q2_path) {
    // Sanity-check upstream ggml's Q2_0 traits match our 64-element / 18-byte codec.
    const auto* q2_traits = ggml_get_type_traits(GGML_TYPE_Q2_0);
    if (q2_traits == nullptr || q2_traits->to_float == nullptr ||
        q2_traits->blck_size != static_cast<int>(kQ2BlockElements) ||
        q2_traits->type_size != kQ2BlockBytes) {
        return Status::Error(ErrorCode::kUnsupported, "GGML_TYPE_Q2_0 traits unavailable or unexpected");
    }
    return MapStatus(si::verify_pq2_to_q2_gguf(pq2_path, q2_path));
}

}  // namespace sonder::backends::llamacpp
