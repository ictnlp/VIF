try:
	from .model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
except ImportError:  # pragma: no cover - optional dependency path
	LlavaLlamaForCausalLM = None
	LlavaConfig = None
