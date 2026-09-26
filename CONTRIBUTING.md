# Contributing

Read [repository instructions](AGENTS.md), the [roadmap](docs/ROADMAP.md), and
[design decisions](docs/DESIGN_DECISIONS.md) first.

## Build and test

- Windows: `powershell -NoProfile -File scripts\build.ps1 -Preset msvc-debug -Test`
- Linux/macOS: `cmake --preset linux-debug && cmake --build --preset linux-debug && ctest --preset linux-debug`

CI runs the `ci-windows` and `ci-linux` presets (warnings as errors). Tests
must pass without network services or model weights; live Ollama checks are
opt-in via `SONDER_TEST_OLLAMA_MODEL` (and optionally `SONDER_TEST_OLLAMA_URL`).

## Rules

- Keep commits focused and use DCO sign-off (`git commit -s`).
- Run `git diff --check`; follow `.editorconfig` and `.gitattributes`.
- The C ABI (`include/sonder_inference.h`) is append-only.
- The project is MIT licensed ([LICENSE](LICENSE)); contributions are accepted
  under the same license. Third-party code requires a record in
  [LICENSE_REVIEW](docs/LICENSE_REVIEW.md) before it is fetched, vendored, or
  linked.
- Never commit model weights, secrets, or large generated outputs.
