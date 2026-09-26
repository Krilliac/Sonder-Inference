// Sonder Inference: loaded model handle.
#pragma once

#include <memory>
#include <string>

#include "sonder/inference/backend.hpp"

namespace sonder::inference {

// Engine-owned handle for a model instance materialized by a backend.
class Model {
public:
    Model(std::string instance_id, std::string backend_name, std::string device_id,
          std::shared_ptr<BackendModel> impl)
        : instance_id_(std::move(instance_id)), backend_name_(std::move(backend_name)),
          device_id_(std::move(device_id)), impl_(std::move(impl)) {}

    [[nodiscard]] const std::string& instance_id() const noexcept { return instance_id_; }
    [[nodiscard]] const std::string& backend_name() const noexcept { return backend_name_; }
    [[nodiscard]] const std::string& device_id() const noexcept { return device_id_; }
    [[nodiscard]] const ModelDescriptor& descriptor() const { return impl_->descriptor(); }
    [[nodiscard]] BackendModel& backend_model() const { return *impl_; }

private:
    std::string instance_id_;
    std::string backend_name_;
    std::string device_id_;
    std::shared_ptr<BackendModel> impl_;
};

}  // namespace sonder::inference
