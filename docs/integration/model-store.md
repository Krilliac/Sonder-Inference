# Model store: serving an existing Ollama library in place

Module: `src/models/` (`SONDER_HAS_MODEL_STORE`), header
`sonder/inference/model_store.hpp`, CLI: `sonder-infer models import-ollama`,
tests: `sonder.model_store.*`. Network is never used and nothing is written.

Direction (2026-11-26): Sonder-Inference becomes the only inference engine and the
local Ollama daemon is retired. The models Ollama already holds (tens of GB per model)
must not be re-downloaded or copied. This module reads Ollama's on-disk library
**read-only and in place** and gives the native backend a name -> files lookup, so
retiring the daemon loses no models.

## What it does and does not do

- Scans an Ollama models directory, resolves `name[:tag]` (and `namespace/name:tag`,
  `host/namespace/name:tag`; default host `registry.ollama.ai`, namespace `library`,
  tag `latest`) to the model GGUF path plus projector, adapter and draft paths.
- Reads the params layer (`num_ctx`, `stop`, `temperature`, ...), the template and system
  layers, the messages layer and the config blob into a typed `ModelRecord`.
- Reports entries SI cannot serve locally, each with a reason code and text:
  Ollama Cloud stubs, safetensors/MLX, missing blobs, truncated blobs, invalid GGUF,
  broken manifests (see "Unsupported entries").
- Verifies blob digests **only on request** (`verify_blobs`, `--verify`).
- Does NOT copy, move, link, rename or delete anything in the store (a test snapshots the
  tree before and after a scan, a verify and a resolve and requires it unchanged).
- Does NOT load a model, start a daemon, or touch the llamacpp backend (see "Backend hook").
- Does NOT persist any configuration. Which store directories are registered is the
  caller's decision (open item below).

## The Ollama on-disk format (verified against upstream source)

Checked 2026-11-26 against `github.com/ollama/ollama` at `main` = `8ae8acb81afba9c3c67adf08190964eb7d1207ef`,
and against a real store (`D:\ollama\models`, read only).

| Fact | Source |
| --- | --- |
| Models dir is `$OLLAMA_MODELS`, default `<home>/.ollama/models` | `envconfig/config.go` `Models()` |
| Blobs live in `<models>/blobs/sha256-<64 hex>`; a digest in a manifest is `sha256:<hex>` (a `-` separator also validates) | `manifest/paths.go` `BlobsPath`, `ValidateDigest` |
| Current manifests: `<models>/manifests-v2/<host>/<namespace>/<model>/<tag>`; `:` in the host is written `%3A`; public hosts `registry.ollama.ai` and `ollama.com` are one registry and v2 stores it as `ollama.com` | `manifest/paths.go` `V2PathForName`, `v2NameFilepath`, `encodeV2ManifestHost`, `canonicalV2Name` |
| Legacy manifests: `<models>/manifests/<host>/<namespace>/<model>/<tag>` (default host `registry.ollama.ai`) | `manifest/paths.go` `LegacyPathForName` |
| Resolution order: v2 first, then legacy (also under the alternate public host) | `manifest/paths.go` `resolveManifestPath` |
| A manifest is `{schemaVersion, mediaType, config, layers[], runner?, format?}`; a layer is `{mediaType, digest, size, from?}` | `manifest/manifest.go` `Manifest`, `manifest/layer.go` `Layer` |
| Media types: `application/vnd.ollama.image.{model,projector,adapter,draft,template,system,params,messages,license,embed,tensor,json}`; `...image.prompt` is the legacy template; config is `application/vnd.docker.container.image.v1+json` | `manifest/layer.go` constants, `server/images.go` `GetModel` |
| `GetModel` consumption: first `model` layer is the model path, further `model` layers are shards; `projector`/`adapter` are lists; `template`/`prompt` -> template; `system` -> string; `params` -> JSON map of options; `messages` -> JSON; `license` -> list; `embed` is deprecated and ignored | `server/images.go` `GetModel` |
| Manifest lists (`application/vnd.ollama.manifest.list.v2+json`): children carry `runner` (`llamacpp`, `ggml`, `mlx`) and `format` (`gguf`, `safetensors`) and are either inline manifests or `{digest}` references to a manifest blob; non-Apple preference is llamacpp, ggml, mlx | `manifest/manifest.go` `selectManifestWithPreferences`, `runnerPreferences`, `resolveLocalManifestReference` |
| Config blob (`ConfigV2`): `model_format`, `model_family(ies)`, `model_type` (parameter size), `file_type` (quantization), `renderer`, `parser`, `requires`, `remote_host`/`remote_model` (cloud), `capabilities`, `context_length` | `types/model/config.go` |
| Params keys and types: `num_ctx num_batch num_gpu main_gpu use_mmap num_thread draft_num_predict num_keep seed num_predict top_k top_p min_p typical_p repeat_last_n temperature repeat_penalty presence_penalty frequency_penalty stop` | `api/types.go` `Options`, `Runner` |
| Name grammar and defaults | `types/model/name.go` `ParseName`, `isValidPart` |
| `<models>/metadata/sha256-<hex>.json` is an Ollama-side cache of GGUF key/values (not part of the model) | `server/gguf_metadata.go` |

Observed in the real store and handled: cloud stubs with `"layers":null` (Go nil slice), a
manifest `config` whose `remote_host` is `https://ollama.com:443`, hf.co four-part names
(`hf.co/<org>/<repo>:<quant>`), models whose model layer carries `from`, a `llamacpp/<64 hex>` model
name, manifests that exist in both trees, and v2-only entries.

The `metadata/` cache is not read: the native backend reads the GGUF itself, and an absent
or stale cache must never decide whether a model is served.

## Unsupported entries

A record has `status` `ready` or `unsupported`; an unsupported record keeps everything that
was parsed and carries `reason` + `detail`. Precedence: the first reason found wins.

| `reason` | Meaning |
| --- | --- |
| `cloud_stub` | config has `remote_host`/`remote_model`: an Ollama Cloud entry with no local weights. It is reached by SI over HTTPS directly, not from the store. |
| `safetensors` | `format`/`model_format` is safetensors, runner `mlx`, tensor layers, or an MLX-only manifest list. SI's native engine loads GGUF. |
| `unsupported_format` | some other non-GGUF `format`. |
| `no_model_layer` | no `image.model` layer and not a cloud stub. |
| `missing_blob` | a model/projector/adapter/draft/text-layer blob is absent from `blobs/`. |
| `blob_size_mismatch` | blob size differs from the manifest (partial or replaced file). |
| `invalid_gguf` | first 24 bytes are not a GGUF header of version 2 or 3 (version 1 is refused the way llama.cpp refuses it). |
| `no_compatible_manifest` | manifest list without a `llamacpp`/`ggml` GGUF child. |
| `manifest_invalid` | unreadable/oversized/not JSON/not an object, malformed layer digest, a list child whose blob does not match its digest. |

A ready record means: every weight blob is present, has the declared size and starts with a
supported GGUF header. It does **not** mean the file loads (see "Open items / UNVERIFIED").

## Digest verification (opt-in)

Scanning reads small files and 24 bytes of each GGUF-class blob. It never hashes a model.
`verify_blobs(record, options)` streams every blob through SHA-256 (64 KiB buffer) and
compares with the blob's digest; `weights_only` skips the small layers; a `CancellationToken`
stops it and `on_blob_start` lets a caller show progress. Verdicts: `ok`, `mismatch`,
`missing`, `io_error`, `cancelled`. Ollama itself only verifies manifest files, so a silently
corrupted model blob is possible in a store and this is the way to find it. Cost: a full read of
every byte (the real store holds about 145 GiB of weights).

## CLI

```
sonder-infer models import-ollama [DIR] [--name NAME] [--verify [--weights-only]] [--json]
```

`DIR` defaults to `$OLLAMA_MODELS`, then `<home>/.ollama/models`. Output is a table (name, status,
size, quantization, parameter size, `num_ctx`, which layers exist, the paths) or, with `--json`,
the `sonder.inference.model_store/1` document (`source`, `models[]` with `files`, `config`,
`params`, `template`, `system`, `gguf`, `warnings`, and `verification` when `--verify` ran).
Exit status: 0 ok, 1 failure (directory unreadable, name not found, verification failed),
2 usage error. Entries that cannot be served are listed, not errors.

## Backend hook (not wired yet)

`ModelRegistry` is the lookup the native llamacpp backend will use to load by name. It is
thread-safe and holds one or more stores (first registered wins):

```cpp
sonder::inference::models::ModelRegistry registry;
registry.add_ollama_dir(dir);                    // scans; no hashing
auto rec = registry.resolve("qwen3.5:4b");       // ready records only
// rec.value().model_path / shard_paths / projector_paths / adapter_paths   -> mmap these in place
// rec.value().params (num_ctx, stop, sampling defaults)                    -> request defaults
// rec.value().template_text / system_prompt / config.renderer             -> prompt rendering
```

`resolve` returns `not_found`, `invalid_argument` (bad name) or `unsupported` (entry exists but
cannot be served; the message names the reason). A miss in the cached scan falls back to a live
lookup, so a model pulled after the last `refresh()` is still found. Nothing in `src/backends`
includes this header yet; the integration is the loader lane's change: resolve the `--model`
name through the registry when it is not an existing file path.

Prompt rendering note for that integration: many current Ollama models have no template layer
(`config.renderer`/`parser` name an Ollama-internal renderer instead) and rely on the GGUF's
`tokenizer.chat_template`. The record exposes `renderer`, `parser` and the template text so the
backend can choose; it does not reimplement Ollama's Go-template or renderers.

## Open items / UNVERIFIED

- Loadability is not verified. `runner` `ggml` entries (all legacy manifests are treated as that by
  Ollama) were written for Ollama's own engine; a few architectures use tensor names or layouts
  that llama.cpp does not read. Only `runner: llamacpp` entries are known to target llama.cpp.
  No model was loaded here (inference services are intentionally down); the loader lane should
  fail clearly per model.
- No configuration persistence: `OLLAMA_MODELS` or an explicit `DIR` selects the store. A durable
  "registered stores" setting (and where the server reads it) is a design choice for the integrator.
- Digest references (`ollama run sha256-<hex>`) and Ollama's `@digest` name suffix are not resolved.
- Manifest files are not themselves verified against their digest (v2 manifests on POSIX are
  symlinks to blobs named by the manifest's SHA-256); list-child blobs are.
- Windows host:port registries in the legacy tree are skipped by Ollama itself; this reader accepts
  them only through the v2 tree (`%3A`).
- Only a synthetic tree is exercised in CI. The real main-PC store (`D:\ollama\models`, which is also
  `OLLAMA_MODELS`) was scanned read-only by hand: 26 entries, 20 ready (145.3 GiB of weights referenced
  in place), 6 unsupported, all Ollama Cloud stubs (two of them with `"layers":null`); no safetensors entry.
