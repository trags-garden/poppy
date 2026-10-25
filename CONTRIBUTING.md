# Contributing

Poppy's source is published openly under the Apache License 2.0
(see [LICENSE](LICENSE)). You are free to read, run, self-host, fork, and modify
it under those terms. For a map of the code, see
[docs/architecture.md](docs/architecture.md).

Poppy is maintained by the Trags team. We are not accepting pull requests right
now. Bug reports and feature requests are welcome as
[GitHub issues](https://github.com/trags-garden/poppy/issues/new/choose), or by
email at feedback@trags.ai. For security issues, see [SECURITY.md](SECURITY.md).

If you build something on top of Poppy, we would love to hear about it.

## Release checklist

1. Bump the version in `pyproject.toml` and both version fields in `server.json`
   together, including existing release metadata.
2. Release through the existing PyPI publish workflow.
3. After the package is live on PyPI, the maintainer runs `mcp-publisher publish`
   from the repository root to update `ai.trags/poppy` in the MCP Registry.
