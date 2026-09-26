# Scaffold status

Created 2026-09-26. **Documentation and structure only.**

## Included

- [Source ownership placeholders](../src/README.md).
- [Future test location](../tests/README.md).
- [Contract workspace](contracts/README.md).
- [Proposed ecosystem boundaries](BOUNDARIES.md).
- Repository editing conventions and contribution instructions.

## Deliberately undecided

Implementation language, build/package system, public API, transport, dependency
versions, license, CI jobs, and deployment model. Existing architecture documents
remain the reference for previously recorded decisions; this scaffold adds no
new runtime promises.

There are no runnable commands, source implementations, mocked services, backend
bindings, model downloads, or runtime tests. Directory names are provisional.

## Next design work

1. Agree repository ownership and the integration boundary with Sonder Runtime.
2. Choose the first supported use case and write its acceptance criteria.
3. Resolve language/toolchain and project/upstream licensing.
4. Review API, lifecycle, cancellation, error, authorization, and budget semantics.
5. Request implementation of a bounded first slice, then add meaningful CI/tests.
