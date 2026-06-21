# Security and collection policy

The crawler and learner operate on untrusted web content.

- Standard HTTP and browser execution only.
- No bypass of authentication, paywalls, DRM, access controls, CAPTCHAs, or robots policies.
- Private and link-local network ranges are blocked unless a local synthetic experiment explicitly enables them.
- Renderer contexts contain no credentials, cookies, cloud metadata access, or host filesystem mounts.
- Content length, redirect count, render time, frame count, and parser output are bounded.
- Raw bytes are hashed before parsing.
- Extracted text cannot invoke tools. Only the host VM emits typed operations.
- Claims require exact supporting spans and are not accepted solely because a generative extractor emitted them.
- Model-generated summaries are caches, never the only evidence stored for a promoted cell.
