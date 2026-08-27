from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class HuggingFaceGeneration:
    """
    One generated sequence from a Hugging Face causal language model.

    token_probabilities contains generated token text paired with the
    probability assigned to that generated token. Callers that do not need
    token probabilities receive an empty list.
    """

    text: str
    token_probabilities: list[tuple[str, float]]


class HuggingFaceRuntime:
    """
    Shared, lazily loaded Hugging Face causal-language-model runtime.

    Instances are cached by model name so generator, decomposer, verifier,
    and scorer adapters can reuse one model/tokenizer pair instead of loading
    the same checkpoint multiple times into memory.
    """

    _instances = {}
    _instances_lock = Lock()

    @classmethod
    def for_model(cls, model_name: str):
        """
        Return the process-wide runtime associated with model_name.
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        normalized_name = model_name.strip()

        with cls._instances_lock:
            if normalized_name not in cls._instances:
                cls._instances[normalized_name] = cls(normalized_name)

            return cls._instances[normalized_name]

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = None
        self._load_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        """
        Report whether the model and tokenizer have already been loaded.
        """
        return self._model is not None and self._tokenizer is not None

    def _ensure_loaded(self):
        """
        Lazily load the tokenizer and causal LM onto the best local device.
        """
        if self.is_loaded:
            return

        with self._load_lock:
            if self.is_loaded:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if torch.cuda.is_available():
                dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                load_kwargs = {
                    "dtype": dtype,
                    "device_map": "auto",
                }
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                dtype = torch.float16
                load_kwargs = {
                    "dtype": dtype,
                }
            else:
                device = torch.device("cpu")
                dtype = torch.float32
                load_kwargs = {
                    "dtype": dtype,
                }

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise ValueError(
                        f"Tokenizer for {self.model_name!r} has neither a "
                        "padding token nor an EOS token."
                    )

                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

            if not torch.cuda.is_available():
                model.to(device)

            model.eval()
            device = model.device

            if model.config.pad_token_id is None:
                model.config.pad_token_id = tokenizer.pad_token_id

            self._tokenizer = tokenizer
            self._model = model
            self._device = device

    def _render_messages(self, messages: list[dict[str, str]]) -> str:
        """
        Render chat messages using the tokenizer's model-specific template.
        """
        self._ensure_loaded()

        if not messages:
            raise ValueError("At least one chat message is required.")

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each chat message must be a dictionary.")

            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(
                    "Each chat message must have role "
                    "'system', 'user', or 'assistant'."
                )

            if not isinstance(message.get("content"), str):
                raise ValueError("Each chat message must contain string content.")

        if not getattr(self._tokenizer, "chat_template", None):
            raise ValueError(
                f"Tokenizer for {self.model_name!r} does not provide a "
                "chat template. Use an instruction-tuned chat model."
            )

        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        n_samples: int = 1,
        max_new_tokens: int = 512,
        return_token_probabilities: bool = False,
    ) -> list[HuggingFaceGeneration]:
        """
        Generate one or more assistant responses.

        Token probabilities are calculated only when requested because
        retaining per-step vocabulary scores increases memory use.
        """
        if temperature < 0:
            raise ValueError("temperature must be non-negative.")

        if n_samples < 1:
            raise ValueError("n_samples must be at least 1.")

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")

        if temperature == 0 and n_samples > 1:
            raise ValueError(
                "n_samples greater than 1 requires temperature > 0 for "
                "Hugging Face generation."
            )

        self._ensure_loaded()

        import torch

        prompt = self._render_messages(messages)

        model_inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(self._device) for key, value in model_inputs.items()
        }

        prompt_length = model_inputs["input_ids"].shape[1]
        do_sample = temperature > 0

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": n_samples,
            "pad_token_id": self._tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": return_token_probabilities,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self._model.generate(
                **model_inputs,
                **generation_kwargs,
            )

        generations = []
        eos_token_ids = self._model.generation_config.eos_token_id

        if eos_token_ids is None:
            eos_token_ids = self._tokenizer.eos_token_id

        if eos_token_ids is None:
            eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        special_token_ids = {
            token_id
            for token_id in [
                self._tokenizer.pad_token_id,
                *eos_token_ids,
            ]
            if token_id is not None
        }

        for sequence_index, sequence in enumerate(output.sequences):
            generated_ids = sequence[prompt_length:]

            text = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            token_probabilities = []

            if return_token_probabilities:
                scores = output.scores

                for step_index, token_id_tensor in enumerate(generated_ids):
                    if step_index >= len(scores):
                        break

                    token_id = int(token_id_tensor.item())

                    if token_id in special_token_ids:
                        continue

                    step_scores = scores[step_index][sequence_index]
                    probability = torch.softmax(
                        step_scores.float(),
                        dim=-1,
                    )[token_id].item()

                    token_text = self._tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )

                    if token_text:
                        token_probabilities.append((token_text, float(probability)))

            generations.append(
                HuggingFaceGeneration(
                    text=text,
                    token_probabilities=token_probabilities,
                )
            )

        return generations
