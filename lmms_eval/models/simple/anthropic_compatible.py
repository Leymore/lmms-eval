import base64
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
import requests as http_requests
from accelerate import Accelerator, DistributedType
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from dotenv import load_dotenv
from loguru import logger as eval_logger
from PIL import Image

from qwen_vl_utils import process_vision_info

load_dotenv(verbose=True)

MODEL_ALIAS_MAP = {
    "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-sonnet-20241022-v2": "anthropic.claude-3-5-sonnet-20241022-v2:0",
}


def get_client_api_key(api_service: str) -> str:
    client_id = os.getenv("AUTH_SERVER_CLIENT_ID", None)
    client_secret = os.getenv("AUTH_SERVER_CLIENT_SECRET", None)
    url = os.getenv("AUTH_SERVER_TOKEN_URL", None)
    assert client_id is not None and client_secret is not None and url is not None, "AUTH_SERVER_CLIENT_ID, AUTH_SERVER_CLIENT_SECRET, and AUTH_SERVER_TOKEN_URL must be set"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    scope_map = {"anthropic": "awsanthropic-readwrite"}
    data = {"grant_type": "client_credentials", "scope": scope_map[api_service]}

    response = http_requests.post(url, headers=headers, data=data, auth=HTTPBasicAuth(client_id, client_secret))
    if response.status_code != 200:
        raise RuntimeError("Error: Could not generate a Bearer API token, please try again")

    return response.json()["access_token"]


@register_model("anthropic_compatible")
class AnthropicCompatible(lmms):
    def __init__(
        self,
        model_version: str = "claude-3.5-sonnet",
        timeout: int = 10,
        max_retries: int = 5,
        max_size_in_mb: int = 20,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        max_frames_num: int = 32,
        batch_size: int = 64,
        use_auth_api: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if not use_auth_api:
            raise NotImplementedError("Only use_auth_api=True is supported for anthropic_compatible")

        self.model_version = model_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_size_in_mb = max_size_in_mb
        self.continual_mode = continual_mode
        self.max_frames_num = max_frames_num
        self.use_auth_api = use_auth_api

        if model_version in MODEL_ALIAS_MAP:
            self.model_name = MODEL_ALIAS_MAP[model_version]
        else:
            self.model_name = model_version

        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        self.bearer_token = get_client_api_key("anthropic")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def refresh_token(self) -> None:
        eval_logger.info("Refreshing API token...")
        self.bearer_token = get_client_api_key("anthropic")
        eval_logger.info("API token refreshed successfully")

    def tok_encode(self, string: str):
        return list(string.encode("utf-8"))

    def tok_decode(self, tokens):
        return ""

    @property
    def eot_token_id(self):
        return 0

    @property
    def rank(self):
        return self._rank

    def encode_image(self, image: Union[Image.Image, str]):
        max_size = self.max_size_in_mb * 1024 * 1024
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        while len(byte_data) > max_size and img.size[0] > 100 and img.size[1] > 100:
            new_size = (int(img.size[0] * 0.75), int(img.size[1] * 0.75))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str, "image/png"

    def encode_video(self, video_path, max_num_frames):
        if max_num_frames == 0:
            return []

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        nframes = max(min(max_num_frames, total_frame_num), 2)

        video_element = {"type": "video", "video": video_path, "nframes": nframes}

        message = [{"role": "user", "content": [video_element]}]
        _, video_inputs = process_vision_info([message])

        if video_inputs is None or len(video_inputs) == 0:
            return []

        video_tensor = video_inputs[0]
        actual_nframes = len(video_tensor)

        if max_num_frames == 1:
            video_tensor = video_inputs[0][:1]
            actual_nframes = 1

        eval_logger.info(f"Number of frames: {actual_nframes}")

        base64_frames = []
        for idx in range(actual_nframes):
            if hasattr(video_tensor, "numpy"):
                frame = video_tensor[idx].numpy()
            elif hasattr(video_tensor[idx], "numpy"):
                frame = video_tensor[idx].numpy()
            else:
                frame = video_tensor[idx]

            if frame.shape[-1] != 3 and frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))

            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append((base64_str, "image/png"))

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        from lmms_eval import utils

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            task = task[0]
            split = split[0]

            batch_payloads = []
            batch_doc_uuids = []
            batch_responses = []

            for i, (context, doc_id_single) in enumerate(zip(contexts, doc_id)):
                doc_uuid = f"{task}___{split}___{doc_id_single}"
                batch_doc_uuids.append(doc_uuid)

                if self.continual_mode is True and self.cache_mode == "resume":
                    if doc_uuid in self.response_cache:
                        response_text = self.response_cache[doc_uuid]
                        if response_text:
                            batch_responses.append(response_text)
                            continue

                visuals = [doc_to_visual[i](self.task_dict[task][split][doc_id_single])]
                if None in visuals:
                    visuals = []
                    imgs = []
                else:
                    visuals = self.flatten(visuals)
                    imgs = []
                    for visual in visuals:
                        if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                            frames = self.encode_video(visual, self.max_frames_num)
                            imgs.extend(frames)
                        elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                            img_data, media_type = self.encode_image(visual)
                            imgs.append((img_data, media_type))
                        elif isinstance(visual, Image.Image):
                            img_data, media_type = self.encode_image(visual)
                            imgs.append((img_data, media_type))

                content = [{"type": "text", "text": context}]
                for img_data, media_type in imgs:
                    content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}})

                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if gen_kwargs["max_new_tokens"] > 4096:
                    gen_kwargs["max_new_tokens"] = 4096

                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": gen_kwargs["max_new_tokens"],
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                }

                batch_payloads.append(payload)
                batch_responses.append(None)

            def process_single_request(payload, i):
                if batch_responses[i] is not None:
                    return batch_responses[i], i

                for attempt in range(self.max_retries):
                    try:
                        correlation_id = str(uuid.uuid4())
                        base_api_url = os.getenv("ANTHROPIC_API_BASE_URL", "https://prod.api.enterprise.internal")
                        url = f"{base_api_url}/llm/v1/aws/model/{self.model_name}/invoke"

                        headers = {
                            "correlationId": correlation_id,
                            "dataClassification": "sensitive",
                            "dataSource": "internet",
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.bearer_token}"
                        }

                        response = http_requests.post(url, headers=headers, json=payload, timeout=120)
                        response.raise_for_status()

                        result = response.json()
                        response_text = result.get("content", [{}])[0].get("text", "")

                        input_text = payload["messages"][0]["content"][0]["text"]
                        eval_logger.info("=" * 64)
                        eval_logger.info(f"Input text: {input_text}")
                        eval_logger.info("-" * 64)
                        eval_logger.info(f"Response text: {response_text}")
                        eval_logger.info("=" * 64)

                        return response_text, i

                    except Exception as e:
                        error_msg = str(e)
                        eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")

                        if "401" in error_msg and "token has expired" in error_msg.lower():
                            eval_logger.info("Token expired, refreshing token and retrying...")
                            try:
                                self.refresh_token()
                                continue
                            except Exception as refresh_error:
                                eval_logger.error(f"Failed to refresh token: {refresh_error}")

                        if attempt == self.max_retries - 1:
                            eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                            return "", i
                        else:
                            time.sleep(self.timeout)

                return "", i

            tasks_to_run = [(payload, i) for i, payload in enumerate(batch_payloads) if batch_responses[i] is None]

            if tasks_to_run:
                max_workers = min(len(tasks_to_run), 32)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {executor.submit(process_single_request, payload, i): i for payload, i in tasks_to_run}

                    for future in as_completed(future_to_index):
                        response_text, i = future.result()
                        batch_responses[i] = response_text

            if self.continual_mode is True:
                for doc_uuid, response_text in zip(batch_doc_uuids, batch_responses):
                    if response_text is not None:
                        self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

            res.extend([r for r in batch_responses if r is not None])
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Anthropic compatible models")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood for Anthropic compatible models")
