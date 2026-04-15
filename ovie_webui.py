import argparse
import math
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "outputs"
SAMPLE_IMAGE_PATH = APP_DIR / "assets" / "sample_image.jpg"
HF_HOME = CACHE_DIR / "HF_HOME"
TORCH_HOME = CACHE_DIR / "TORCH_HOME"

for directory in (CACHE_DIR, OUTPUT_DIR, HF_HOME, TORCH_HOME):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_HOME))
os.environ.setdefault("GRADIO_TEMP_DIR", str(OUTPUT_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from models.models import OVIEModel  # noqa: E402
from utils.pose_enc import extri_intri_to_pose_encoding  # noqa: E402
from utils.utils import center_crop_arr  # noqa: E402


MODEL_REVISION = "v1.0"
MODEL_LOCK = threading.Lock()
MODEL = None
MPS_RETRY_HINTS = (
    "mps",
    "metal",
    "placeholder storage",
    "not implemented",
    "not supported",
)

DEFAULT_CAMERA = {
    "yaw": 28.0,
    "pitch": 12.0,
    "distance": 2.4,
}

CAMERA_PRESETS = {
    "Safe": {
        "yaw": (-35.0, 35.0),
        "pitch": (-20.0, 20.0),
        "distance": (1.5, 3.0),
    },
    "Experimental": {
        "yaw": (-90.0, 90.0),
        "pitch": (-45.0, 45.0),
        "distance": (0.75, 5.0),
    },
}
DEFAULT_CAMERA_PRESET = "Safe"

DEFAULT_STATUS = (
    "Ready. Release a slider to render a new view automatically. "
    "Uploading or resetting an image will also render. "
    "The first run may take a while while OVIE downloads."
)


def get_preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_built() and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return "CUDA"
    if device.type == "mps":
        return "MPS"
    return "CPU"


def should_retry_on_cpu(exc: Exception) -> bool:
    if DEVICE.type != "mps":
        return False

    message = str(exc).lower()
    return any(hint in message for hint in MPS_RETRY_HINTS)


def switch_device(device: torch.device) -> None:
    global DEVICE, MODEL

    if DEVICE.type == device.type:
        return

    DEVICE = device
    with MODEL_LOCK:
        MODEL = None


DEVICE = get_preferred_device()

CUSTOM_CSS = """
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto;
    padding: 32px 20px 56px;
    background:
        radial-gradient(circle at top, rgba(0, 0, 0, 0.055), transparent 28%),
        linear-gradient(180deg, #fafafa 0%, #ffffff 28%);
}

body {
    background:
        linear-gradient(180deg, #fafafa 0%, #ffffff 24%),
        repeating-linear-gradient(
            90deg,
            rgba(0, 0, 0, 0.025) 0,
            rgba(0, 0, 0, 0.025) 1px,
            transparent 1px,
            transparent 48px
        );
}

.page-shell {
    gap: 1.5rem;
}

.surface-panel {
    border: 1px solid #ebebeb;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.92);
    box-shadow:
        0 22px 50px rgba(0, 0, 0, 0.055),
        0 1px 0 rgba(255, 255, 255, 0.95) inset;
    overflow: hidden;
}

.surface-panel > .wrap {
    border: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
}

.surface-panel .label-wrap {
    margin-bottom: 0.6rem;
}

.surface-panel .block-title {
    font-size: 1.2rem;
    letter-spacing: -0.03em;
}

.panel-note {
    margin-top: 0.95rem;
    color: #666666;
    font-size: 0.94rem;
    line-height: 1.6;
}

.control-panel .form {
    gap: 0.9rem;
}

.control-panel .wrap {
    padding-top: 0.5rem;
}

.status-copy {
    margin-top: 1rem;
    padding: 0.95rem 1rem;
    border: 1px solid #ececec;
    border-radius: 16px;
    background: #fbfbfb;
    color: #4f4f4f;
    line-height: 1.55;
}

.image-frame,
.result-frame {
    overflow: hidden;
}

.image-frame img,
.result-frame img {
    border-radius: 20px;
}

.action-row {
    gap: 0.75rem;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 20px 14px 36px;
    }
}
"""


def load_sample_image() -> Image.Image:
    return Image.open(SAMPLE_IMAGE_PATH).convert("RGB")


def get_model(progress: gr.Progress | None = None) -> OVIEModel:
    global MODEL

    with MODEL_LOCK:
        if MODEL is None:
            if progress is not None:
                progress(0.1, desc="Downloading/loading OVIE from Hugging Face")
            MODEL = OVIEModel.from_pretrained(
                "kyutai/ovie",
                revision=MODEL_REVISION,
            ).to(DEVICE)
            MODEL.eval()
            if progress is not None:
                progress(0.5, desc="OVIE ready")

    return MODEL


def describe_view(yaw_deg: float, pitch_deg: float) -> tuple[str, str]:
    if abs(yaw_deg) < 6:
        horizontal = "front view"
        camera_note = "close to the original camera position"
    elif yaw_deg > 0:
        horizontal = "left view" if abs(yaw_deg) < 34 else "farther-left view"
        camera_note = "to the left of the subject"
    else:
        horizontal = "right view" if abs(yaw_deg) < 34 else "farther-right view"
        camera_note = "to the right of the subject"

    if abs(pitch_deg) < 5:
        vertical = "at eye level"
        height_note = "at roughly the same height"
    elif pitch_deg > 0:
        vertical = "from slightly above" if abs(pitch_deg) < 16 else "from higher above"
        height_note = "slightly above the subject"
    else:
        vertical = "from slightly below" if abs(pitch_deg) < 16 else "from lower below"
        height_note = "slightly below the subject"

    return f"{horizontal}, {vertical}", f"The camera will move {camera_note} and {height_note}."


def restore_default_view():
    yaw = DEFAULT_CAMERA["yaw"]
    pitch = DEFAULT_CAMERA["pitch"]
    distance = DEFAULT_CAMERA["distance"]
    return yaw, pitch, distance


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def apply_camera_preset(
    preset_name: str,
    yaw_deg: float,
    pitch_deg: float,
    distance: float,
):
    preset = CAMERA_PRESETS.get(preset_name, CAMERA_PRESETS[DEFAULT_CAMERA_PRESET])
    yaw_min, yaw_max = preset["yaw"]
    pitch_min, pitch_max = preset["pitch"]
    distance_min, distance_max = preset["distance"]
    return (
        gr.update(
            minimum=yaw_min,
            maximum=yaw_max,
            value=clamp(yaw_deg, yaw_min, yaw_max),
        ),
        gr.update(
            minimum=pitch_min,
            maximum=pitch_max,
            value=clamp(pitch_deg, pitch_min, pitch_max),
        ),
        gr.update(
            minimum=distance_min,
            maximum=distance_max,
            value=clamp(distance, distance_min, distance_max),
        ),
    )


def build_camera_token(
    image_size: int,
    yaw_deg: float,
    pitch_deg: float,
    distance: float,
) -> torch.Tensor:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    x = -distance * math.sin(yaw) * math.cos(pitch)
    y = distance * math.sin(pitch)
    z = -distance * math.cos(yaw) * math.cos(pitch)

    extrinsics = torch.tensor(
        [[[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z]]],
        device=DEVICE,
        dtype=torch.float32,
    )
    dummy_intrinsics = torch.zeros(1, 1, 3, 3, device=DEVICE, dtype=torch.float32)
    camera = extri_intri_to_pose_encoding(
        extrinsics=extrinsics.unsqueeze(0),
        intrinsics=dummy_intrinsics,
        image_size_hw=(image_size, image_size),
    )
    return camera[..., :7].squeeze(0)


def _generate_view(
    input_image: Image.Image | None,
    yaw_deg: float,
    pitch_deg: float,
    distance: float,
    progress: gr.Progress = gr.Progress(),
    allow_cpu_retry: bool = True,
):
    try:
        progress(0.0, desc="Preparing OVIE")
        model = get_model(progress)

        if input_image is None:
            input_image = load_sample_image()
            source_name = "sample image"
        else:
            source_name = "uploaded image"

        image_size = model.image_size
        progress(0.6, desc="Preparing input image")
        img_pil = center_crop_arr(input_image.convert("RGB"), image_size)
        img_tensor = ToTensor()(img_pil).unsqueeze(0).to(DEVICE)

        cam_token = build_camera_token(image_size, yaw_deg, pitch_deg, distance)

        progress(0.8, desc="Generating novel view")
        pred_tensor = model(x=img_tensor, cam_params=cam_token)
        pred_display = pred_tensor[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        pred_image = Image.fromarray((pred_display * 255).astype(np.uint8))

        output_path = OUTPUT_DIR / f"ovie-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        pred_image.save(output_path)

        device_name = get_device_name(DEVICE)
        view_title, _ = describe_view(yaw_deg, pitch_deg)
        status = (
            f"Generated {view_title} from {source_name} on {device_name}. "
            f"Model cache: {HF_HOME}. Saved: {output_path.name}"
        )
        return pred_image, status
    except Exception as exc:
        if allow_cpu_retry and should_retry_on_cpu(exc):
            progress(0.05, desc="MPS failed, retrying on CPU")
            switch_device(torch.device("cpu"))
            pred_image, status = _generate_view(
                input_image,
                yaw_deg,
                pitch_deg,
                distance,
                progress,
                allow_cpu_retry=False,
            )
            if pred_image is not None:
                return pred_image, f"{status} MPS execution failed, so the launcher retried on CPU."

        return None, f"Generation failed on {get_device_name(DEVICE)}: {exc}"


@torch.inference_mode()
def generate_view(
    input_image: Image.Image | None,
    yaw_deg: float,
    pitch_deg: float,
    distance: float,
    progress: gr.Progress = gr.Progress(),
):
    return _generate_view(input_image, yaw_deg, pitch_deg, distance, progress)


@torch.inference_mode()
def generate_view_ui(
    input_image: Image.Image | None,
    yaw_deg: float,
    pitch_deg: float,
    distance: float,
    progress: gr.Progress = gr.Progress(),
):
    pred_image, status = _generate_view(input_image, yaw_deg, pitch_deg, distance, progress)
    return pred_image, status


def create_demo() -> gr.Blocks:
    default_preset = CAMERA_PRESETS[DEFAULT_CAMERA_PRESET]
    with gr.Blocks(title="OVIE") as demo:
        with gr.Column(elem_classes=["page-shell"]):
            with gr.Row(equal_height=True):
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["surface-panel"]):
                        input_image = gr.Image(
                            value=load_sample_image(),
                            type="pil",
                            show_label=False,
                            sources=["upload", "clipboard"],
                            elem_classes=["image-frame"],
                        )
                        with gr.Row(elem_classes=["action-row"]):
                            sample_button = gr.Button("Use Sample Image", variant="secondary")
                            reset_button = gr.Button("Reset Camera", variant="secondary")
                        gr.Markdown(
                            "Choose a range preset, then release a slider to render. Uploading a new image or resetting the camera also renders automatically.",
                            elem_classes=["panel-note"],
                        )

                    with gr.Group(elem_classes=["surface-panel", "control-panel"]):
                        camera_preset = gr.Radio(
                            choices=list(CAMERA_PRESETS),
                            value=DEFAULT_CAMERA_PRESET,
                            label="Range Preset",
                            info="Safe stays closer to OVIE's typical viewpoints. Experimental widens the camera bounds for stress testing.",
                        )
                        yaw = gr.Slider(
                            minimum=default_preset["yaw"][0],
                            maximum=default_preset["yaw"][1],
                            value=DEFAULT_CAMERA["yaw"],
                            step=1,
                            label="Yaw",
                            info="Positive moves the camera left. Negative moves it right.",
                        )
                        pitch = gr.Slider(
                            minimum=default_preset["pitch"][0],
                            maximum=default_preset["pitch"][1],
                            value=DEFAULT_CAMERA["pitch"],
                            step=1,
                            label="Pitch",
                            info="Positive moves the camera higher. Negative moves it lower.",
                        )
                        distance = gr.Slider(
                            minimum=default_preset["distance"][0],
                            maximum=default_preset["distance"][1],
                            value=DEFAULT_CAMERA["distance"],
                            step=0.05,
                            label="Distance",
                            info="Higher values place the camera farther away from the subject.",
                        )

                with gr.Column(scale=7):
                    with gr.Group(elem_classes=["surface-panel"]):
                        output_image = gr.Image(
                            type="pil",
                            interactive=False,
                            show_label=False,
                            elem_classes=["result-frame"],
                        )
                        status = gr.Markdown(
                            value=DEFAULT_STATUS,
                            elem_classes=["status-copy"],
                        )

            generation_inputs = [input_image, yaw, pitch, distance]
            generation_outputs = [output_image, status]
            generation_kwargs = {
                "fn": generate_view_ui,
                "inputs": generation_inputs,
                "outputs": generation_outputs,
                "trigger_mode": "always_last",
                "concurrency_limit": 1,
                "concurrency_id": "generation",
                "api_visibility": "private",
            }

            api_button = gr.Button(visible=False)
            api_button.click(
                fn=generate_view,
                inputs=generation_inputs,
                outputs=[output_image, status],
                api_name="generate_view",
                concurrency_limit=1,
                concurrency_id="generation",
            )

            demo.load(**generation_kwargs)

            sample_button.click(
                fn=load_sample_image,
                inputs=[],
                outputs=input_image,
                queue=False,
                api_visibility="private",
            ).then(**generation_kwargs)

            camera_preset.change(
                fn=apply_camera_preset,
                inputs=[camera_preset, yaw, pitch, distance],
                outputs=[yaw, pitch, distance],
                queue=False,
                api_visibility="private",
            ).then(**generation_kwargs)

            reset_button.click(
                fn=restore_default_view,
                inputs=[],
                outputs=[yaw, pitch, distance],
                queue=False,
                api_visibility="private",
            ).then(**generation_kwargs)

            input_image.change(**generation_kwargs)

            for slider in (yaw, pitch, distance):
                slider.release(**generation_kwargs)

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = create_demo()
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        theme=gr.themes.Monochrome(
            font=("ui-sans-serif", "system-ui", "sans-serif"),
            font_mono=("IBM Plex Mono", "ui-monospace", "Consolas", "monospace"),
        ),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
