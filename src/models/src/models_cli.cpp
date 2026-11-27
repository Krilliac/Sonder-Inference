// `sonder-infer models import-ollama` (docs/integration/model-store.md).
//
// Read-only: lists what an existing Ollama models directory would serve through SI's
// native engine, without starting any daemon, copying any file or writing to the store.
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "sonder/inference/model_store.hpp"
#include "store_text.hpp"

namespace sonder::inference::models {

namespace fs = std::filesystem;

namespace {

constexpr int kExitOk = 0;
constexpr int kExitFailure = 1;
constexpr int kExitUsage = 2;

const char* kUsage =
    "usage: sonder-infer models import-ollama [DIR] [--name NAME] [--verify [--weights-only]] [--json]\n"
    "\n"
    "Lists what the Ollama models directory DIR would serve through SI's native engine. DIR defaults\n"
    "to $OLLAMA_MODELS, then <home>/.ollama/models. Nothing is started, copied or written: model\n"
    "files stay where Ollama put them and are referenced in place.\n"
    "\n"
    "  --name NAME      only this model (name[:tag], namespace/name:tag; default tag latest)\n"
    "  --verify         also re-hash the blobs of every ready model and compare with the manifest\n"
    "                   digest. An incomplete entry cannot be verified: reported FAILED (exit 1).\n"
    "                   Opt-in: reads every byte (minutes for multi-GB files); progress on stderr\n"
    "  --weights-only   with --verify, hash only model/projector/adapter/draft blobs\n"
    "  --json           print the sonder.inference.model_store/2 document instead of the table\n"
    "\n"
    "Entries that cannot be served locally (Ollama Cloud stubs, safetensors/MLX, missing or truncated\n"
    "blobs, invalid GGUF) are listed with the reason. Exit status: 0 ok, 1 failure (directory\n"
    "unreadable, name not found, verification failed), 2 usage error.\n";

int usage_error(std::ostream& err, const std::string& message) {
    err << "error: models: " << message << "\n" << kUsage;
    return kExitUsage;
}

std::string human_bytes(std::uint64_t n) {
    static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(n);
    std::size_t u = 0;
    while (v >= 1024.0 && u < 4) {
        v /= 1024.0;
        ++u;
    }
    char buf[32];
    if (u == 0) {
        std::snprintf(buf, sizeof buf, "%llu B", static_cast<unsigned long long>(n));
    } else {
        std::snprintf(buf, sizeof buf, "%.1f %s", v, kUnits[u]);
    }
    return buf;
}

std::string pad(std::string s, std::size_t width) {
    if (s.size() < width) s.append(width - s.size(), ' ');
    return s;
}

constexpr std::size_t kMaxPathText = 8192;
constexpr std::size_t kMaxFieldText = 128;

// For the operator's terminal only (never an HTTP response), escaped with the store's
// sanitizer (store_text.hpp): a POSIX path may hold newlines, ESC or invalid UTF-8, and any
// path may hold bidi controls. Does not throw on a Windows name with an unpaired surrogate.
std::string path_text(const fs::path& p) {
    try {
        const auto u = p.generic_u8string();
        return store_text::printable(std::string_view(reinterpret_cast<const char*>(u.data()), u.size()), kMaxPathText);
    } catch (const std::exception&) {
        return "<path not representable as UTF-8>";
    }
}

// An untrusted config field (quantization, parameter size) for the table.
std::string field_text(const std::string& s) { return s.empty() ? std::string("-") : store_text::printable(s, kMaxFieldText); }

struct Options {
    std::optional<std::string> dir;
    std::optional<std::string> name;
    bool verify = false;
    bool weights_only = false;
    bool json = false;
};

// Returns an error message, empty on success.
std::string parse_options(const std::vector<std::string>& args, Options& o) {
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--verify") {
            o.verify = true;
        } else if (a == "--weights-only") {
            o.weights_only = true;
        } else if (a == "--json") {
            o.json = true;
        } else if (a == "--name" || a.rfind("--name=", 0) == 0) {
            std::string value;
            if (a == "--name") {
                if (i + 1 >= args.size()) return "--name needs a value";
                value = args[++i];
            } else {
                value = a.substr(7);
            }
            if (o.name) return "--name given more than once";
            o.name = std::move(value);
        } else if (a.rfind("--", 0) == 0) {
            return "unknown option " + a;
        } else {
            if (o.dir) return "unexpected argument '" + a + "'";
            o.dir = a;
        }
    }
    if (o.weights_only && !o.verify) return "--weights-only requires --verify";
    return {};
}

const char* verdict_label(BlobVerdict v) {
    return to_string(v);
}

}  // namespace

int models_main(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    if (args.empty()) return usage_error(err, "expected a subcommand (import-ollama)");
    if (args[0] == "--help" || args[0] == "-h") {
        out << kUsage;
        return kExitOk;
    }
    if (args[0] != "import-ollama") return usage_error(err, "unknown subcommand '" + args[0] + "'");
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-h") {
            out << kUsage;
            return kExitOk;
        }
    }
    Options o;
    if (const std::string msg = parse_options(args, o); !msg.empty()) return usage_error(err, msg);

    const fs::path dir = o.dir ? fs::path(*o.dir) : default_ollama_models_dir();
    if (dir.empty()) {
        return usage_error(err, "no DIR given and neither OLLAMA_MODELS nor a home directory is set");
    }
    // A listing tool: weights are not hashed unless --verify asks for it (verify_blobs below),
    // so resolving one --name does not read a multi-GB file by surprise.
    StoreOptions store_options;
    store_options.verify_weights = WeightVerification::off;
    auto store = OllamaStore::open(dir, store_options);
    if (!store.ok()) {
        err << "error: models: '" << path_text(dir) << "': " << store.status().message() << "\n";
        return kExitFailure;
    }

    ScanResult scan;
    if (o.name) {
        auto one = store.value().resolve(*o.name);
        if (!one.ok()) {
            err << "error: models: " << one.status().message() << " (" << path_text(store.value().root()) << ")\n";
            return one.status().code() == ErrorCode::invalid_argument ? kExitUsage : kExitFailure;
        }
        scan.records.push_back(std::move(one).value());
    } else {
        scan = store.value().scan();
    }

    // Opt-in verification (hashes every selected ready model's blobs).
    std::vector<std::optional<VerifyReport>> reports(scan.records.size());
    bool verify_failed = false;
    if (o.verify) {
        for (std::size_t i = 0; i < scan.records.size(); ++i) {
            const ModelRecord& rec = scan.records[i];
            // An unsupported entry has nothing to hash. An incomplete one (layers dropped by
            // ScanLimits::max_retained_text_bytes) is passed on: verify_blobs() refuses it, so
            // it is reported as FAILED (exit 1), never skipped and never counted as verified.
            if (rec.status != EntryStatus::ready && rec.status != EntryStatus::incomplete) continue;
            VerifyOptions vo;
            vo.weights_only = o.weights_only;
            vo.on_blob_start = [&](const BlobRef& b) {
                err << "verifying " << rec.name.display() << " " << to_string(b.role) << " " << b.digest.substr(0, 19)
                    << " (" << human_bytes(b.disk_size.value_or(b.declared_size)) << ")\n";
            };
            reports[i] = verify_blobs(rec, vo);
            if (!reports[i]->ok()) verify_failed = true;
        }
    }

    if (o.json) {
        json::Value doc = to_json(scan, store.value().root());
        json::Array& models = doc.object()["models"].array();
        for (std::size_t i = 0; i < models.size(); ++i) {
            if (!reports[i]) continue;
            json::Array blobs;
            for (const auto& b : reports[i]->blobs) {
                blobs.emplace_back(json::Object{{"role", to_string(b.blob.role)},
                                                {"digest", b.blob.digest},
                                                {"verdict", verdict_label(b.verdict)},
                                                {"detail", b.detail}});
            }
            models[i].object().set("verification", json::Object{{"ok", reports[i]->ok()},
                                                                {"inventory_complete", reports[i]->inventory_complete},
                                                                {"refusal", reports[i]->refusal},
                                                                {"blobs", std::move(blobs)}});
        }
        out << doc.dump() << "\n";
        return verify_failed ? kExitFailure : kExitOk;
    }

    std::size_t ready = 0;
    std::size_t incomplete = 0;
    std::size_t shared_entries = 0;
    std::uint64_t ready_bytes = 0;
    std::map<std::string, std::string> weight_owner;  // weight blob digest -> first ready entry that references it
    out << "Ollama models directory: " << path_text(store.value().root()) << " (read-only; nothing is copied)\n";
    for (std::size_t i = 0; i < scan.records.size(); ++i) {
        const ModelRecord& r = scan.records[i];
        out << pad(r.name.display(), 52) << " ";
        if (r.ready()) {
            ++ready;
            // Count each weight blob once: entries can share blobs (a manifest-list child and the
            // legacy manifest that points at the same file, or tags built on one base model).
            std::string shares_with;
            for (const auto& b : r.blobs) {
                if (b.role != BlobRole::model && b.role != BlobRole::projector && b.role != BlobRole::adapter &&
                    b.role != BlobRole::draft) {
                    continue;
                }
                const auto [it, inserted] = weight_owner.emplace(b.digest, r.name.display());
                if (inserted) {
                    ready_bytes = b.declared_size > UINT64_MAX - ready_bytes ? UINT64_MAX : ready_bytes + b.declared_size;
                } else if (b.role == BlobRole::model && shares_with.empty()) {
                    shares_with = it->second;
                }
            }
            if (!shares_with.empty()) ++shared_entries;
            out << "ready        " << pad(human_bytes(r.weights_bytes), 10) << " "
                << pad(field_text(r.config.quantization), 8) << " " << pad(field_text(r.config.parameter_size), 7);
            if (r.params.num_ctx) out << " num_ctx=" << *r.params.num_ctx;
            if (!r.projector_paths.empty()) out << " +projector";
            if (!r.adapter_paths.empty()) out << " +adapter";
            if (r.template_text) out << " +template";
            if (r.system_prompt) out << " +system";
            out << "\n";
            out << "    model: " << path_text(r.model_path) << "\n";
            for (const auto& p : r.shard_paths) out << "    shard: " << path_text(p) << "\n";
            for (const auto& p : r.projector_paths) out << "    projector: " << path_text(p) << "\n";
            for (const auto& p : r.adapter_paths) out << "    adapter: " << path_text(p) << "\n";
            for (const auto& p : r.draft_paths) out << "    draft: " << path_text(p) << "\n";
            if (!shares_with.empty()) out << "    note: same weights file as " << shares_with << "\n";
        } else if (r.status == EntryStatus::incomplete) {
            ++incomplete;
            out << "incomplete   (some small layers were not read: ScanLimits::max_retained_text_bytes was spent; "
                   "resolve reads them)\n";
        } else {
            out << "unsupported  (" << to_string(r.reason) << ") " << r.detail << "\n";
        }
        for (const auto& w : r.warnings) out << "    warning: " << w << "\n";
        if (reports[i]) {
            out << "    verify: " << (reports[i]->ok() ? "ok" : "FAILED") << " (" << reports[i]->blobs.size()
                << " blobs)\n";
            if (!reports[i]->inventory_complete) out << "      not verified: " << reports[i]->refusal << '\n';
            for (const auto& b : reports[i]->blobs) {
                if (b.verdict != BlobVerdict::ok) {
                    out << "      " << to_string(b.blob.role) << " " << b.blob.digest << ": "
                        << verdict_label(b.verdict) << (b.detail.empty() ? "" : " - " + b.detail) << "\n";
                }
            }
        }
    }
    for (const auto& n : scan.notes) out << "note: " << n << "\n";
    out << scan.records.size() << " entries: " << ready << " ready (" << human_bytes(ready_bytes)
        << " of distinct weight files referenced in place";
    if (shared_entries != 0) out << "; " << shared_entries << " entries share a weights file with an earlier one";
    out << "), " << (scan.records.size() - ready - incomplete) << " unsupported";
    if (incomplete != 0) out << ", " << incomplete << " incomplete";
    out << "\n";
    if (!o.verify) {
        out << "digests not verified (pass --verify to hash the blobs; it reads every byte)\n";
    }
    return verify_failed ? kExitFailure : kExitOk;
}

}  // namespace sonder::inference::models
