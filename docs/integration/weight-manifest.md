# Weight-content digest manifests

Module: `src/artifacts/` (`SONDER_HAS_WEIGHT_MANIFEST`), CLI: `sonder-infer weights`,
tests: `sonder.artifacts.*`. Network is never used.

Very large models ship as multi-file safetensors or GGUF sets of 100 GB or more.
A flipped bit in the middle of a shard is otherwise never noticed. A manifest
records the SHA-256 of every file so a later check can find it.

## What this does and does not establish

- It is evidence about bytes at rest in one directory at the moment of hashing or
  verifying. It supports the remaining ROADMAP item "associate inspected content
  with loaded bytes" but does not close it: nothing binds the verified bytes to what
  a backend later maps (the file can change in between), and a manifest made from a
  corrupt copy certifies the corrupt copy unless `--expected` is used.
- Provenance comes only from `expected_sha256`, which is as trustworthy as the
  saved upstream listing it was taken from.
- Not certified: tensor-extent validity, loadability, directory immutability.

## Manifest format (version 1)

One JSON document, written compact with a trailing newline:

```json
{"format":"sonder.weight_manifest","version":1,"model_dir":"my-model",
 "created_at":"2026-11-19T08:00:00Z","tool":"sonder-infer","tool_version":"0.1.0",
 "files":[{"path":"model-00001-of-00002.safetensors","size":10000,
           "sha256":"<64 lowercase hex>","expected_sha256":"<optional, 64 lowercase hex>"}]}
```

- `path`: relative, `/`-separated, sorted bytewise, unique. Refused on read and write:
  absolute paths, empty/`.`/`..` segments, backslashes, `:` (drive letters, NTFS
  streams), control characters, segments ending in `.` or space.
- `model_dir` is the directory's leaf name, informational only (no absolute path is stored).
- `expected_sha256` is present only for files covered by `--expected`.
- A different `version` is rejected rather than guessed at.

## Commands

```
sonder-infer weights manifest DIR -o FILE [--expected HF_TREE_JSON] [--quiet]
sonder-infer weights verify DIR MANIFEST [--sample N [--seed S]] [--full] [--quiet]
```

`manifest` hashes every file below DIR. An existing `-o` file is replaced only if it is
itself a weight manifest (regenerating in place works, and a manifest inside DIR is not
hashed); a link, a directory, a special file or any other file at that path, including a
weight shard, is refused with exit 4 before any hashing and again right before the
write. The check is not atomic (see TOCTOU below). `--expected` takes one file: a Hugging Face
tree-API response saved locally (an array of
`{"type":"file","path":...,"lfs":{"oid":<sha256>}}`) or a model-API object with
`siblings`. The tree endpoint is non-recursive and paginated by default, so save it with
`?recursive=true`, follow the `cursor` pages (default `limit` 1000), and concatenate
every page into a single JSON array before passing it in. Only LFS oids (and explicit `sha256` members) are used; the git blob `oid`
of a non-LFS file is a SHA-1 and is ignored. A computed digest that differs from the
expected one is reported (`expected-mismatch:`), the manifest is still written, and
the exit status is 2. An expected path that is absent from DIR is reported as
`missing:` (exit 3).

`verify` always checks, without hashing: every manifest file is present, its byte
size, files in DIR that the manifest does not list (`extra:`), links and special
entries (`unsafe:`), and each entry's `expected_sha256` against its `sha256`. Content
is hashed only with `--sample N` (up to N files chosen by a seeded, platform-independent
shuffle; `--seed S` repeats a run, otherwise a seed is drawn and printed) or `--full`.
The choice is drawn from all manifest entries so a seed selects the same indices
whatever is on disk; files already reported missing or size-mismatched are not hashed,
so fewer than N may be.
The default is therefore a cheap structure check that cannot see a same-size bit flip.

Exit status: 0 ok, 1 I/O failure, 2 mismatch (size, sha256, expected_sha256),
3 missing or extra file (or unsafe entry), 4 usage error or invalid manifest, 130
cancelled. A mismatch outranks a missing/extra file when both occur. Progress
(`hashing PATH: done/total`) goes to stderr, at most one line per 2 seconds plus one
per finished file; results go to stdout.

## Safety rules

- The directory walk never follows links. Symlinks, Windows reparse points
  (junctions, cloud placeholders), special files and names that cannot be represented
  as a safe relative path (including names that are not valid UTF-8, or on Windows have
  no UTF-8 form, such as unpaired surrogates) are `unsafe`. `manifest` refuses a directory containing one
  (exit 4), `verify` reports it (exit 3) and never reads through it. A Hugging Face
  cache snapshot directory is symlinks into `blobs/`, so point the tool at a real
  directory instead. DIR itself may be a link.
- Names are untrusted: every path printed by the CLI (stdout, stderr, progress) goes
  through `printable_path()`, which writes backslashes, control characters, C1 controls,
  bidi and line-separator code points and ill-formed bytes as `\\`, `\xNN` or `\uXXXX`,
  so a file name cannot forge an output line or a terminal escape. `DirListing::unsafe`
  and `VerifyReport::unsafe` keep the raw names for a supervisor. C1 controls
  (U+0080-U+009F) are also refused as path content.
- Every manifest path is re-validated on read and every component is checked for a
  link before the file is opened.
- Hashing uses one buffer (default 1 MiB, clamped 4 KiB to 64 MiB) per file; memory
  does not grow with file size.
- Check-then-open races (TOCTOU) are not closed: a hostile writer racing the tool can
  still swap a file after the link check.

## Library API

`sonder/inference/weight_manifest.hpp`, namespace `sonder::inference::weights`:
`build_manifest`, `parse_manifest`, `manifest_to_json`, `verify_manifest`,
`hash_file`, `list_directory`, `parse_expected_json`, `select_sample`,
`validate_relative_path`, `weights_main`. All are reentrant and hold no global state;
progress callbacks and `CancellationToken` polling happen on the calling thread, once
per buffer. Supervisor integration point: see `INTEGRATION_NOTES.md`.
