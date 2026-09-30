#include "doctest/doctest.h"
#include "sonder/inference/model_architecture.hpp"

using sonder::inference::ModelArchitecture;
using sonder::inference::classify_model_architecture;

TEST_CASE("GGUF architecture metadata classifies attention-only models") {
    CHECK(classify_model_architecture({{"general.architecture", "llama"}}) == ModelArchitecture::attention_only);
    CHECK(classify_model_architecture({{"general.architecture", "qwen2"}, {"qwen2.attention.layer_count", "32"}}) ==
          ModelArchitecture::attention_only);
}

TEST_CASE("GGUF architecture metadata classifies recurrent models") {
    CHECK(classify_model_architecture({{"general.architecture", "mamba"}}) == ModelArchitecture::recurrent);
    CHECK(classify_model_architecture({{"general.architecture", "custom"}, {"custom.ssm_state_size", "16"}}) ==
          ModelArchitecture::recurrent);
}

TEST_CASE("GGUF architecture metadata classifies hybrid models") {
    CHECK(classify_model_architecture({{"general.architecture", "jamba"}}) == ModelArchitecture::hybrid);
    CHECK(classify_model_architecture({{"general.architecture", "qwen3_next"}}) == ModelArchitecture::hybrid);
    CHECK(classify_model_architecture({{"general.architecture", "custom"},
                                       {"custom.ssm_state_size", "16"},
                                       {"custom.attention.layer_count", "4"}}) == ModelArchitecture::hybrid);
    CHECK(classify_model_architecture({{"general.architecture", "qwen3.8"},
                                       {"qwen3.8.block_count", "64"},
                                       {"qwen3.8.attention.layer_count", "16"},
                                       {"qwen3.8.block.0.deltanet", "1"},
                                       {"qwen3.8.block.0.attention.layer_count", "16"}}) ==
          ModelArchitecture::hybrid);
    CHECK(classify_model_architecture({{"general.architecture", "qwen3.8"},
                                       {"qwen3.8.attention.head_count", "0"},
                                       {"qwen3.8.block.0.deltanet", "1"}}) == ModelArchitecture::hybrid);
    CHECK(classify_model_architecture({{"general.architecture", "custom"},
                                       {"custom.attention.head_count", "0"},
                                       {"custom.block.0.deltanet", "1"}}) == ModelArchitecture::recurrent);
    CHECK(classify_model_architecture({{"general.architecture", "custom"},
                                       {"custom.attention.head_count", "4"},
                                       {"custom.block.0.deltanet", "1"}}) == ModelArchitecture::hybrid);
}

TEST_CASE("GGUF architecture metadata recognizes recurrent families") {
    CHECK(classify_model_architecture({{"general.architecture", "rwkv7"}}) == ModelArchitecture::recurrent);
    CHECK(classify_model_architecture({{"general.architecture", "rwkv6qwen2"}}) == ModelArchitecture::recurrent);
}
