# Upstream License and Dependency Review

This repository's research catalog intentionally includes projects under different licenses and governance models.

Do **not** assume that being public/open source means code can be copied into Sonder Inference under whatever license Sonder eventually uses.

Before adding an upstream code dependency or copying an implementation:

1. identify exact repository and revision;
2. read the license at that revision;
3. check submodule/subdependency licenses;
4. distinguish linking, dynamic loading, process boundary, and copied code;
5. preserve notices/attribution where required;
6. check model/weight licenses separately from engine code;
7. record the decision here or in an ADR.

For research papers, implement concepts from the paper/design independently unless a deliberate code-dependency decision is made.

## Dependency record template

```markdown
### <dependency>
- repository:
- revision/tag:
- evaluated:
- license:
- intended use:
- linkage/process boundary:
- notices required:
- security/maintenance notes:
- approved:
```

No license conclusions from the initial broad web sweep are considered authoritative until verified against the exact adopted revision.
