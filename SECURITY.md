# Security

The ZharfAI team takes security seriously and will work quickly to resolve security issues in this library.

## Reporting a vulnerability

If you find a security vulnerability, please do not open a public issue. Report it by email to security@zharf.ai instead, and give us reasonable time to investigate and fix it before you disclose it publicly.

Please include:
- A description of the vulnerability
- Steps to reproduce it
- Your assessment of its impact
- Any mitigations you know of

## Keeping your key safe

- Keep your API key (`zk-…`) in the `ZHARF_API_KEY` environment variable or a secrets manager, never in source code or in a repository.
- Give each app its own key, and set a spending limit on each key in the platform dashboard.
- Revoke a key at once if it may have leaked, and create a new one.
- Keep this library up to date.

## Contact

For any other security question, write to security@zharf.ai.
