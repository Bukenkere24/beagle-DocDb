# beagle-DocDb
AI-powered DocDB/ETL Helper for CortexOne that converts raw documents into structured, chunked, metadata-rich, embeddings-ready payloads.

# Platform Notes

## 1. API Request Payload Size

There is currently **no documented request payload size limit** in the official documentation. Treat the limit as **undefined rather than unlimited**, and avoid designing your application around very large single requests.

## 2. API Rate Limits

There are currently **no documented API rate limits** in the official documentation. Since the limits are undefined, it is recommended to implement reasonable safeguards such as:

* Request throttling
* Retry mechanisms with exponential backoff
* Queueing for high-volume workloads

## 3. CLI / Git Deployment

A **CLI or Git-based deployment workflow is not currently available**.

The supported deployment workflow is:

```text
Studio → Tool Editor → Publish
```

Deployment requires manually pasting the code into the Tool Editor before publishing.

## 4. Environment Secrets

To use environment variables and secrets:

1. Create secrets under `/user/secrets`.
2. Attach the required secrets to the tool in the Tool Editor.
3. Access the secrets in Python using:

```python
import os

api_key = os.environ.get("KEY_NAME")
```

## 5. Versioning

Published versions are **immutable**. Any modification to the code, configuration, or dependencies requires creating and publishing a **new version** of the tool.

---

**Note:** Because several platform limits (payload size and rate limits) are currently undocumented, applications should be designed defensively and avoid making assumptions about unrestricted usage.
