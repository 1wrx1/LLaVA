from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import os

model_path = "liuhaotian/llava-v1.5-13b"

os.environ['TRANSFORMERS_CACHE'] = '/grp01/saas_lqqu/runxi/llava/.cache'

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    cache_dir='/grp01/saas_lqqu/runxi/llava/.cache'
)