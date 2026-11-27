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
- Pins the store root when it is opened and opens every file relative to it without following
  links, checks small layers (config, template, system, params, messages) against their size and
  SHA-256 on every read, hashes weights on first use by default, and can hand a loader the very
  handles it verified (see "Trust and verification").
- A full report of every blob digest is still a separate call (`verify_blobs`, `--verify`).
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
| `unsafe_path` | a manifest, a directory on its path, `blobs/` or a blob is a symlink, junction or other reparse point, is not a regular disk file, or (manifests and small layers) has more than one hard link. |
| `digest_mismatch` | a small layer, or a weight that was hashed (see the policy below), does not hash to its digest. |
| `blob_busy` | (Windows) a weight blob is open for writing by another process, so it cannot be opened with writers denied to hash or serve it. Transient: the registry checks again on the next lookup. |
| `hash_limit` | the weights were not hashed to the end: a blob is larger than `StoreOptions::max_weight_file_bytes` (default 1 TiB), the distinct weight blobs of the resolve add up to more than `max_resolve_weight_bytes` (default 2 TiB; refused before anything is opened or hashed), or hashing ran past `weight_hash_timeout` (default 3 hours per resolve) or was cancelled (`weight_hash_cancel`, or `ResolveOptions::cancel` of that call). Transient: nothing is remembered and the next lookup tries again. |
| `no_compatible_manifest` | manifest list without a `llamacpp`/`ggml` GGUF child. |
| `manifest_invalid` | unreadable/oversized/not JSON/not an object, a repeated JSON key, a size that is not a non-negative integer (`24.9`, `-1`, `"24"`), a digest that is not exactly `sha256:<64 hex>` (the `sha256-` spelling is only a blob file name), more than 64 content layers or 64 list children, small layers over 4 MiB each or 16 MiB together, declared sizes that overflow 64 bits, a list child whose blob does not match its digest. |

A ready record means: every blob it needs is a regular file inside the store with the declared
size, every small layer matched its SHA-256, and every weight starts with a supported GGUF header;
`weight_check` says how the weights were checked against their digests (`hashed_now`,
`cached_identity` or `unchecked`; `weights_verified()` is true only for `hashed_now`). It does
**not** mean the file loads (see "Open items / UNVERIFIED").

## Trust and verification

The threat model is a library another local user or process may be able to write to. What the
reader enforces (security reviews of PR 135, three rounds). "Verified" below means only this:
the bytes were hashed in this call, through a handle that denies writers on Windows, and matched.

- **The root is pinned.** `OllamaStore::open` opens the store root once and keeps that directory
  handle (shared by copies of the store and by every record read through it, `ModelRecord::anchor`).
  A link at the root path itself is followed there, once (the operator chose it). Every later open,
  probe and directory listing starts from that handle, so retargeting or renaming the root path
  after `open()` cannot move `resolve()`, `scan()`, `revalidate()` or `verify_blobs()` to another
  directory (test "retargeting the root link after open() does not move the store"). `root()` and
  the record paths keep the path for display only.
- **No links, no escape.** Every manifest, list child, config, text layer and blob is opened once
  through `src/common/safe_file.hpp` (shared with the weight manifests): one path segment at a time,
  each relative to the directory handle opened before it, never following a symlink, junction or
  other reparse point. Windows: `NtCreateFile` with `OBJECT_ATTRIBUTES::RootDirectory` (Win32 has no
  handle-relative open), `FILE_OPEN_REPARSE_POINT`, an attribute check of every handle and a
  final-path check of every component against `<root's current final path>\segment` (which also
  refuses 8.3 aliases). POSIX: `openat` + `O_NOFOLLOW`. The leaf must be a regular disk file. Size,
  hash and content all come from that one handle with a bounded read (cap + 1 bytes), so a file
  swapped or grown between check and read is caught. Directories are listed from their own handles
  too (Windows `GetFileInformationByHandleEx(FileIdBothDirectoryInfo)`, POSIX `fdopendir` on a
  private descriptor with `fstatat(AT_SYMLINK_NOFOLLOW)` per entry), streamed one entry at a time
  into a fixed buffer; directory links under `manifests*/` are skipped with a note, and `blobs/`
  itself must be a real directory. Windows opens each directory with `FILE_OPEN_REPARSE_POINT` but
  without `FILE_DIRECTORY_FILE` (Microsoft does not list the two as compatible) and checks the
  directory and reparse attributes on the handle it got.
- **Manifest symlinks (POSIX only, `manifests-v2/` only).** Ollama writes each v2 name as a relative
  symlink to `blobs/sha256-<manifest hex>` except on Windows (`manifest/manifest.go`
  `WriteManifest`). Such a leaf is read with `readlinkat` and never followed, opened or `stat`ed
  through: its text must be a relative path that, resolved lexically from the link's directory, is
  exactly `blobs/sha256-<64 lowercase hex>` (absolute targets, `.` or empty segments, climbing above
  the root, other directories and other names are refused as `unsafe_path`). That blob is then
  opened from the pinned root, the link is `lstat`ed again (same identity and text, or the read is
  refused as a retarget), and the blob's bytes must hash to `<hex>` (Ollama's own rule in
  `openManifestFileLocked`). A link under the legacy `manifests/` tree is refused. On Windows every
  manifest link is refused: Ollama writes copies there, and following a link could open a UNC target.
- **Hard links.** Manifests and small layers must have one link: a hard link to a file outside the
  store would otherwise turn a template or system layer into a read of that file (and its digest
  check into a guessing oracle). Weights may have more than one: Ollama hard-links split-model
  shards into `blobs/.ollama-split-*/` while it runs one (`llm/model_split.go`), so refusing them
  would break real stores. Such a weight gets a warning and is vouched for only by its SHA-256
  (under `off` it is served on size and header alone, as every weight is there).
- **Small layers** (config, template/prompt, system, params, messages, list children) are checked
  against the declared size and SHA-256 on every read; a mismatch makes the entry unsupported,
  never a silently dropped prompt.
- **Weights.** Size (from the handle) and the GGUF header are checked on every read. The full
  SHA-256 follows the weight policy, and every weight hash reads through a handle that on Windows
  is opened without `FILE_SHARE_WRITE`/`FILE_SHARE_DELETE`: while it is open no other process can
  write, truncate, rename or delete the file, and while another process holds it open for writing
  the open is refused and the entry is reported `blob_busy` (transient; it is not hashed or served
  until the writer closes it).
  - `first_use` (default for `resolve()` and the registry's `resolve()`): each weight is hashed
    once; the digest found (match or mismatch) is remembered per store by the file's identity
    (volume serial + 128-bit `FILE_ID_INFO` file id on Windows, `st_dev`/`st_ino` on POSIX, size,
    hard-link count, last-write and change time), so an unchanged file is not re-read and a changed
    one is. A remembered digest is not a hash of the current bytes: such a record says
    `WeightCheck::cached_identity`, never `hashed_now`, and `weights_verified()` is false for it.
    On such a hit the identity is read again from the same open handle after the lookup; a
    difference or a failed query fails the entry at once (transient), it is not hashed instead.
    Concurrent first uses of one identity share a single hash: one thread hashes (its record says
    `hashed_now`), the others wait for its result (`cached_identity`) and retry if it failed. A file
    whose identity is weak (`FileStamp::weak_id`, the 64-bit fallback below) is hashed on every
    resolve: its digest is never looked up nor remembered (test "a weak file identity is hashed on
    every first_use resolve and never remembered", through a test hook that marks every identity
    weak). `scan()`/`list()` do not hash (`unchecked`).
  - `always`: every resolve and every scan hashes (`hashed_now`), nothing is remembered.
  - `off`: never (`unchecked`; the CLI listing uses this, `--verify` still hashes on request). Not
    suitable for a library that untrusted local writers can modify.
  At most `kMaxConcurrentWeightHashes` (2) hashes run at once per opened root: the weight checks and
  `verify_blobs()` of records read through it share the slots (copies of the store share them
  too). Only the hash read loop runs inside a slot; no hash runs while a cache or registry lock is
  held. `on_weight_hash` runs before a weight hash, on the thread that wants to hash, while that
  thread holds nothing of the store: no slot and no single-flight claim (under `first_use` the
  claim is taken only after the callback returns, so nobody ever waits for a callback; several
  threads needing one unchanged file may each call back, and one of them then hashes it). It may
  therefore wait for other threads that use a store or registry (test "an on_weight_hash callback
  may wait for another thread that resolves the same model"). On its own thread, though, every
  `resolve()`, `resolve_for_load()` and registry `resolve()` fails at once with `invalid_state`,
  `revalidate()` returns false and `scan()` hashes nothing, so a callback cannot recurse into
  itself (test "an on_weight_hash callback that calls back into the store fails fast instead of
  deadlocking").
  Hashing is bounded three ways (`StoreOptions`), so one hostile manifest cannot hold a resolve
  and the two hash slots for days. (1) The declared sizes of the distinct weight blobs of one
  resolve may add up to `max_resolve_weight_bytes` (default 2 TiB); more fails before any file is
  opened, hashed or announced (test "weights that add up past max_resolve_weight_bytes fail
  before anything is hashed"; a layer repeated under one digest counts once). (2) A file larger
  than `max_weight_file_bytes` (default 1 TiB) is not hashed. (3) All the hashing of one resolve
  (and of one `verify_blobs()` call) has one deadline, `weight_hash_timeout`, default 3 hours
  (`kDefaultWeightHashTimeout`: 2 TiB at the portable SHA-256's slow end, about 90 minutes per TiB),
  counted from when it starts to hash and including the wait for a hash slot or for another
  thread's hash; zero removes it (a service should keep a finite value). A hash also stops when
  the store-wide `weight_hash_cancel` is cancelled (a shutdown switch: never resettable, the
  options are copied at `open()`) or when the `ResolveOptions::cancel` given to that one
  `resolve()` / `resolve_for_load()` is, which also ends the wait for a slot or for another
  thread's hash and affects nothing else (test "a per-call cancel stops one resolve and leaves
  the store usable"). Whichever limit is hit, the entry is `hash_limit`, a transient failure:
  nothing is remembered, a registry never reuses it, and threads waiting for that hash are
  released to try for themselves (test "weight hashing is bounded by file size, time and
  cancellation"). `verify_blobs()` applies `max_weight_file_bytes`, the deadline and
  `weight_hash_cancel` of the store the record came from (test "verify_blobs applies ...").
- **What `hashed_now` and `VerifiedWeights` mean.** `hashed_now`: every weight blob, opened while
  the record was produced or revalidated, hashed to its digest through that handle in that call.
  It says nothing about the file once that handle is closed. A loader that must load exactly the
  hashed bytes uses `OllamaStore::resolve_for_load` / `ModelRegistry::resolve_for_load`, which
  return those same open handles in a move-only `VerifiedWeights` (one per distinct weight digest,
  `find(digest)`, each with its own `check`), and maps or reads those handles; it must not reopen
  `model_path` & co. `resolve_for_load` hashes every weight through the returned handles by
  default (`always`, whatever the store's policy, `off` included: test "resolve_for_load hashes by
  default even on a store opened with verification off"); `verified()` is true only when every
  file is `hashed_now`. A caller that passes `first_use` explicitly accepts `cached_identity`. On
  Windows the handles deny writers for as long as the `VerifiedWeights` lives.
  `verified()` means "hashed through this handle in this call": a statement about a past hash,
  not a guarantee that the bytes cannot change afterwards. POSIX cannot deny writers, and Windows
  cannot revoke a writable mapping that existed before the open (Known limits below).
- **Cached records are revalidated.** `ModelRegistry::resolve` passes every cached ready record
  (from the scan or from an earlier live lookup) through `OllamaStore::revalidate`: each source
  file and blob must still have the identity it had when it was read; otherwise the name is
  resolved again. Every record carries the files it was built from, also when it failed: the
  manifest's stamp once it opened (so a hard link removed, a file shrunk or rewritten is seen), the
  manifest symlink's own stamp, every manifest-list child (its stamp, or `absent` when it was not
  there), and a `failed` marker for anything that could not be opened or read (an I/O error, a busy
  blob, a change mid-read, a refused open). A cached unsupported record (from the scan, or a scan
  entry that a later resolve found broken) is returned again only while every stamped file is
  unchanged, every absent one and every blob it could not open is still absent, and nothing
  failed; otherwise the name is resolved again. So a repaired blob or list child, a finished pull,
  a removed hard link or a released writer is served without `refresh()`, and an unchanged corrupt
  multi-GB blob is not re-hashed on every request. Successful live lookups are cached (at most
  1024; `refresh()` drops them). `refresh()` keeps the records of a store that `add_ollama_dir()`
  registered while it ran. `list()` returns the registry's records as shared read-only pointers,
  not copies.
- **Bounds.** Manifests and list children 4 MiB, 64 content layers, 64 list children, small layers
  4 MiB each and 16 MiB per manifest, each digest validated once per manifest, `params.stop` at most
  64 entries of at most 1024 bytes (a later params layer replaces `stop`, as Ollama's map decode
  does), byte totals added with overflow checks, at most 64 warnings and 256 scan notes, and a name
  longer than the longest valid one (`kMaxOllamaNameBytes`, 601 bytes) is refused before it is split.
  Directory enumeration (`StoreOptions::limits`, `ScanLimits`): at most 4096 entries are read from
  any one directory, 65536 per scan or lookup in total, and 1024 manifests per scan; the caps are
  applied while entries are read, before anything is kept, and a scan that reaches one says so in
  its notes. The case-insensitive lookup streams the directories on the name's path, keeps no
  entries, and stops at the first match; its nested listings share the one `max_scan_entries`
  count, checked before and after each nested listing, so a lookup reads at most that many
  entries in all (test "a name lookup that misses reads at most max_scan_entries entries across
  nested directories").
  Retained content (`ScanLimits::max_retained_text_bytes`, default 64 MiB): the small layers
  (config, params, template, system, messages) one scan keeps, all records together. Each
  layer's declared size is charged before it is read; once the next one would not fit, the scan
  reads no more of them, notes it, and sets `ModelRecord::text_omitted` on the records concerned.
  Those layers are opened, stamped and checked for size and link count (one that cannot be opened,
  or has the wrong size, makes the record unsupported, as reading it would) but not read or
  hashed, so the config, params and text fields of such a record, and what is classified from
  them, are empty or provisional. `resolve()` reads everything, `OllamaStore::revalidate()`
  returns false for a `text_omitted` record, and the registry never serves a cached
  `text_omitted` record without resolving it again. What a record keeps of a layer is also
  capped per field: a config string 1 KiB (cut), a config array 32 entries of 256 bytes, params
  unmodelled options ("extras") 64 entries and 64 KiB in all, `runner` / `format` / a layer's
  `from` 256 bytes; anything over is cut or skipped with a warning (test "config strings and
  arrays and params extras are capped per field"). The registry counts all of it (template,
  system, messages, config strings and arrays, `params.stop`, `params.extra`, runner, format,
  parent) in one running total over the scan records of all its stores plus its live results,
  and strips what would not fit (`text_omitted`). `add_ollama_dir()` and `refresh()` hand each
  store's scan what is left of that one allowance and retain each store's records before the
  next store is read, so N stores never hold N budgets (test "refresh and add_ollama_dir share
  one retained-content allowance across stores"). Not counted, because each is bounded per record
  by a constant (about 100 KiB together): names, paths, the blob list, source stamps, `detail`
  and warnings; so a library of `max_models` records holds at most about that much more.
  A scan that runs out of memory for one entry reports that entry as unsupported
  (`manifest_invalid`, never reused) and goes on. Tests "config and params layers are charged to
  max_retained_text_bytes before they are read" and "a scan keeps template, system and messages
  text only within max_retained_text_bytes".
- **Names.** Besides Ollama's part rules, a part (the host included) may not end in `.` and may not
  be a Windows device name (`CON`, `PRN`, `AUX`, `NUL`, `COM0-9`, `LPT0-9`, with or without an
  extension or port), on every platform; Win32 would otherwise open `coder` for `coder.`. Directory
  entries whose names are not representable as UTF-8 (unpaired UTF-16 surrogates) are skipped with a
  note instead of throwing.
- **Messages.** `Status` messages, `detail`, `warnings` and scan notes quote untrusted text at most
  128 bytes through one sanitizer (`src/models/src/store_text.hpp`): C0 controls, DEL, every byte that
  is not part of well-formed UTF-8 (raw `0x80`-`0xFF`, overlongs, surrogates, cut sequences) and the
  bytes of C1 controls, bidi/zero-width marks, line and paragraph separators, BOM and interlinear
  annotation controls are escaped as `\xNN`, and a backslash is doubled. Messages never name the store's absolute path (notes
  are store-relative; `not_found` says "is not in the store"), and never include a cloud stub's
  `remote_host` (the JSON document strips URL credentials from it). The CLI table prints paths and
  config fields through the same sanitizer; nothing it prints is meant to become an HTTP response.

Known limits (the remaining post-check windows):

- **POSIX has no deny-write.** A process allowed to write a blob can change its bytes after the
  hash, including while a loader maps it (pages not yet faulted in read the new bytes), even
  through a `VerifiedWeights` handle. Where writers of the store are untrusted, make the blobs
  read-only to them (ownership/ACLs); `always` re-hashes on every resolve but still cannot close
  the window between the hash and the mapping.
- **Windows deny-write does not cover** a writable section that another process mapped before the
  open (and kept after closing its handle) or raw volume access by an administrator.
- **Remembered digests trust metadata between checks.** Under `first_use`, a writer who rewrites a
  blob in place with the same size and link count and restores `LastWriteTime` and `ChangeTime`
  (Windows lets anyone with `FILE_WRITE_ATTRIBUTES` set both) is not seen until the next real hash;
  a rewrite within the file system's timestamp granularity looks unchanged too. On POSIX `st_ctime`
  cannot be set back by a user. Such a record says `cached_identity`, never `hashed_now`, and
  `resolve_for_load` does not take that path unless asked to (`first_use`). Use `always` where it
  matters.
- **Record paths are not pinned.** `model_path`, `shard_paths` and the other paths name files under
  the root path given to `open()`; reopening them (instead of using `VerifiedWeights`) gets whatever
  is there by then, including another tree if the root link was retargeted.
- **File systems that do not implement `FileIdInfo`** (`GetFileInformationByHandleEx(FileIdInfo)`
  fails with `ERROR_INVALID_PARAMETER`, `ERROR_INVALID_FUNCTION` or `ERROR_NOT_SUPPORTED`) fall back
  to the 64-bit `BY_HANDLE_FILE_INFORMATION` index, and the identity is marked weak
  (`FileStamp::weak_id`): such a file is never matched against a remembered digest, it is hashed
  every time. Any other `FileIdInfo` failure fails the open. Which error a given file system
  returns has not been tested here (NTFS implements the class).
- **Writers are blocked while a blob is held (Windows).** A hash, a `verify_blobs()` read or a live
  `VerifiedWeights` keeps the blob open without `FILE_SHARE_WRITE`/`FILE_SHARE_DELETE`, so for that
  time `ollama rm`, a prune or a re-pull that rewrites or deletes that blob fails with a sharing
  violation on Ollama's side. A loader that keeps `VerifiedWeights` for the life of a loaded model
  keeps those blobs from being removed for that long.
- **Records belong to the store that read them.** `revalidate()` refuses a record from another
  `OllamaStore`, even one opened on the same directory (records are tied to the root handle that
  read them); the registry only revalidates records against the store that produced them.
- **Enumeration caps hide entries.** A directory over `max_dir_entries` is read only partly (which
  entries depends on the file system's order), and a store over the per-scan caps is listed only
  partly; the notes say so, and `resolve()` of an exact name does not enumerate. Which records
  keep their text under `max_retained_text_bytes` follows the scan's name order.
- **One resolve can still read up to 16 MiB of text** (the per-manifest small-layer cap) while it
  runs; the retained-text budget bounds what scans and the registry keep, not a single call.
- **Hash limits are per resolve, not per caller.** `weight_hash_timeout` is one deadline for all
  the hashing of one resolve, and `max_resolve_weight_bytes` caps its declared weights, but
  threads that retry after a `hash_limit` start a new resolve with a new deadline; a service
  should rate-limit them. Request-level admission for an HTTP path is not here; the store is not
  wired to the server.

## Full digest report (opt-in)

`verify_blobs(record, options)` streams each distinct blob (by digest) through SHA-256 from a
no-follow handle opened relative to the record's pinned root (one 1 MiB buffer per call; on
Windows the handle denies writers, so a blob another process has open for writing is reported
`io_error`, and each hash takes one of the root's hash slots), checks the file did not change while
it was hashed, and compares with the digest; `weights_only` skips the small layers; a
`CancellationToken` stops it and `on_blob_start` lets a caller show progress. Verdicts: `ok`,
`mismatch`, `missing`, `unsafe` (a link or not a regular file; not read), `io_error`, `cancelled`.
Ollama checks a blob's digest when it pulls it but does not re-check it when it loads a model, so a
blob that was corrupted after the pull goes unnoticed and this (or the `first_use` policy) is the
way to find it. Cost: a full read of every byte (the real store holds about 145 GiB of weights).

## CLI

```
sonder-infer models import-ollama [DIR] [--name NAME] [--verify [--weights-only]] [--json]
```

`DIR` defaults to `$OLLAMA_MODELS`, then `<home>/.ollama/models`. Output is a table (name, status,
size, quantization, parameter size, `num_ctx`, which layers exist, the paths) or, with `--json`,
the `sonder.inference.model_store/2` document (`source`, `models[]` with `files`, `config`,
`params`, `template`, `system`, `gguf`, `weights_verified`, `weight_check`, `text_omitted`,
`warnings`, and `verification` when `--verify` ran). Schema `/2` (PR 135 round 4) replaced `/1`:
`weights_verified` now means `weight_check == "hashed_now"` (hashed in this call), and
`weight_check` and `text_omitted` were added.
Exit status: 0 ok, 1 failure (directory unreadable, name not found, verification failed),
2 usage error. Entries that cannot be served are listed, not errors.

## Backend hook (not wired yet)

`ModelRegistry` is the lookup the native llamacpp backend will use to load by name. It is
thread-safe and holds one or more stores (first registered wins):

```cpp
sonder::inference::models::ModelRegistry registry;
registry.add_ollama_dir(dir);                          // pins the root, scans; no hashing
auto m = registry.resolve_for_load("qwen3.5:4b");      // ready records only; hashes through the
                                                       // returned handles (m.value().weights.verified())
// m.value().weights.files()[i].native_handle (HANDLE / fd)        -> map THESE handles; never
//                                                                    reopen record.model_path
// m.value().record.model_path / shard_paths / projector_paths    -> which file is which (display)
// m.value().record.params (num_ctx, stop, sampling defaults)     -> request defaults
// m.value().record.template_text / system_prompt / config.renderer -> prompt rendering
```

`resolve` / `resolve_for_load` return `not_found`, `invalid_argument` (bad name) or `unsupported`
(entry exists but cannot be served; the message names the reason). A miss in the cached scan falls
back to a live lookup, so a model pulled after the last `refresh()` is still found. Every cached
ready record is revalidated against the disk before it is returned, a cached unsupported one only
while its reason still holds. `resolve` checks weights per the registry's policy (default
`first_use`, `ModelRegistry(StoreOptions)` chooses); `resolve_for_load` hashes them through the
handles it returns unless the caller passes another policy (see "Trust and verification").
`VerifiedWeights` owns its handles (move-only,
closed on destruction; on Windows they keep writers out for as long as it lives), so the loader
keeps it alive for as long as the model is mapped. File positions are unspecified after the hash:
map the handles or use positioned reads. Nothing in `src/backends` includes this header yet; the
integration is the loader lane's change: resolve the `--model` name through the registry when it is
not an existing file path, and load from the handles (llama.cpp's own loader opens paths, so that
lane needs a handle- or memory-based load path; until then the "Record paths are not pinned" limit
applies).

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
- A regular-file manifest has no digest to check it against (Ollama treats it the same way); a
  symlinked v2 manifest (POSIX) and list-child blobs are hash-checked.
- Windows host:port registries in the legacy tree are skipped by Ollama itself; this reader accepts
  them only through the v2 tree (`%3A`). A name whose host contains `:` therefore never builds a
  legacy path (on Windows `c:5000` would read as a drive-relative path outside the store), every
  candidate path is also checked to stay lexically under the store root, and the anchored open
  refuses a segment containing `:` on Windows.
- Draft layers: Ollama keeps the first `image.draft` layer as the draft model and the later ones as
  its shards; `draft_paths` lists all of them in manifest order, first = draft model.
- Weights that several entries reference (a manifest-list child and the legacy manifest that points
  at the same file) are counted once in the CLI total, and the later entry is noted as sharing them.
- Only a synthetic tree is exercised in CI. The real main-PC store (`D:\ollama\models`, which is also
  `OLLAMA_MODELS`) was scanned read-only by hand: 26 entries, 20 ready (145.3 GiB, summed per entry by
  the earlier CLI total, which counted a blob shared by two entries twice; the CLI now counts it once
  and that scan was not repeated), 6 unsupported, all Ollama Cloud stubs (two of them with `"layers":null`); no safetensors entry.
