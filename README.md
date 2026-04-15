# OVIE

A Pinokio launcher for the official [kyutai-labs/ovie](https://github.com/kyutai-labs/ovie) repository.

## What This Launcher Does

- Clones the upstream OVIE repository into `app/`
- Installs the locked Python environment with `uv sync --frozen`
- Launches a real local web UI instead of exposing a notebook
- Downloads `kyutai/ovie` from Hugging Face automatically on first generate and caches it locally
- Saves generated images into `outputs/`

## How To Use

1. Click `Install`.
2. Click `Start`.
3. Wait for Pinokio to switch to `Open Web UI`.
4. Upload an image or use the bundled sample image.
5. Adjust `Yaw`, `Pitch`, and `Distance`. Releasing any slider automatically renders the new view.
6. Use `Outputs` to browse saved renders.
7. Use `Update` to pull the latest launcher and upstream repository changes.
8. Use `Reset` to remove the cloned app and its installed environment.

## Programmatic Access

The launcher web UI exposes the inference action as the Gradio endpoint `generate_view`.

### JavaScript

Use the Gradio JavaScript client against the running local app:

```javascript
import { Client, handle_file } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:<PORT>/");
const result = await client.predict("/generate_view", [
  await handle_file("input.jpg"),
  32,
  12,
  2.4
]);
console.log(result.data);
```

### Python

Call the same endpoint from Python:

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:<PORT>/")
result = client.predict(
    handle_file("input.jpg"),
    32,
    12,
    2.4,
    api_name="/generate_view",
)
print(result)
```

### Curl

You can call the generated Gradio API directly:

```bash
curl -X POST "http://127.0.0.1:<PORT>/gradio_api/call/generate_view" \
  -H "Content-Type: application/json" \
  -d "{\"data\":[\"app/assets/sample_image.jpg\",32,12,2.4]}"
```

## Notes

- The first inference may take a while because the launcher downloads `kyutai/ovie` from Hugging Face and warms the model.
- The launcher prefers `CUDA`, then `MPS` on Apple Silicon, then `CPU`. If an `MPS`-specific runtime error occurs during inference, it retries on `CPU`.
- The camera controls map to the same translation recipe used by the original notebook example.
- The UI renders automatically on page load, on image changes, and whenever you release a camera slider.
- The upstream training and preprocessing scripts in `app/` still include CUDA-specific paths and should be treated as NVIDIA-first.
- The upstream repository still includes the original notebooks in `app/`, but the launcher now exposes a normal web UI by default.
