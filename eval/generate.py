import json
import os
import torch
from datasets import load_dataset
from PIL import Image
from diffusers import CogView4Pipeline, FluxPipeline, SanaPipeline, Lumina2Pipeline
from transformers import AutoTokenizer, T5EncoderModel
from infinity.models.infinity import Infinity
from infinity.models.basic import *
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from janus.models import MultiModalityCausalLM, VLChatProcessor
from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5
from transformers import Emu3ForConditionalGeneration, AutoProcessor

def validate_message_fmt(dataset):
    for row in dataset:
        assert "messages" in row, "Each row must have a 'messages' key"
        assert "task_id" in row, "Each row must have a 'task_id' key"
        for message in row["messages"]:
            assert isinstance(message, dict), "Each message must be a dictionary"
            assert "role" in message, "Each message must have a 'role' key"
            assert "content" in message, "Each message must have a 'content' key"
            assert message["role"] in ["user", "assistant"], "Role must be 'user' or 'assistant'"

def generate_cogview(prompt, output_path):
    pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    image = pipe(
        prompt=prompt,
        guidance_scale=3.5,
        num_images_per_prompt=1,
        num_inference_steps=50,
        width=1024,
        height=1024,
    ).images[0]
    
    image.save(output_path)

def generate_flux(prompt, output_path, model_name="flux-dev", device="cuda", offload=False):
    device = torch.device(device)
    is_schnell = model_name == "flux-schnell"
    
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model(model_name, device="cpu" if offload else device)
    ae = load_ae(model_name, device="cpu" if offload else device)

    opts = SamplingOptions(
        prompt=prompt,
        width=1024,
        height=1024,
        num_steps=4 if is_schnell else 50,
        guidance=3.5,
        seed=42,
    )

    x = get_noise(
        1,
        opts.height,
        opts.width,
        device=device,
        dtype=torch.bfloat16,
        seed=opts.seed,
    )

    timesteps = get_schedule(
        opts.num_steps,
        x.shape[-1] * x.shape[-2] // 4,
        shift=(not is_schnell),
    )

    if offload:
        t5, clip = t5.to(device), clip.to(device)
    inp = prepare(t5=t5, clip=clip, img=x, prompt=opts.prompt)

    if offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(device)

    x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    x = unpack(x.float(), opts.height, opts.width)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    if offload:
        ae.decoder.cpu()
        torch.cuda.empty_cache()

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    img.save(output_path)

def generate_janus(prompt, output_path):
    model_path = "deepseek-ai/Janus-1.3B"
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    ).to(torch.bfloat16).cuda().eval()

    conversation = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag

    with torch.inference_mode():
        generated_image = vl_gpt.generate_image(
            prompt,
            temperature=1,
            parallel_size=1,
            cfg_weight=5,
            image_token_num_per_image=576,
            img_size=384,
            patch_size=16,
        )
        generated_image.save(output_path)

def generate_infinity(prompt, output_path, model_path, vae_path):
    device = torch.device('cuda')
    
    vae = vae_model(
        vae_path, 
        "dynamic",
        codebook_dim=32,
        codebook_size=2**32,
        patch_size=16,
        encoder_ch_mult=[1, 2, 4, 4, 4],
        decoder_ch_mult=[1, 2, 4, 4, 4],
        test_mode=True
    ).to(device)

    infinity = Infinity(
        vae_local=vae,
        text_channels=2048,
        text_maxlen=512,
        depth=32,
        embed_dim=2048,
        num_heads=16,
        drop_path_rate=0.1,
        mlp_ratio=4,
        block_chunks=8
    ).to(device)
    
    infinity.load_state_dict(torch.load(model_path))
    infinity.eval()

    text_tokenizer = AutoTokenizer.from_pretrained("t5-large")
    text_encoder = T5EncoderModel.from_pretrained("t5-large")
    
    scale_schedule = dynamic_resolution_h_w[1.0]["1M"]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            generated_image = infinity.generate(
                prompt=prompt,
                text_tokenizer=text_tokenizer,
                text_encoder=text_encoder,
                scale_schedule=scale_schedule,
                cfg_scale=3.0,
                temperature=1.0
            )
            
    generated_image.save(output_path)

def generate_sana(prompt, output_path, model_name="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"):
    pipe = SanaPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)

    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.5,
        num_inference_steps=20,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )[0]

    image[0].save(output_path)

def generate_emu3(prompt, output_path, model_id="BAAI/Emu3-Chat-hf"):
    model = Emu3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs_dict = processor.apply_chat_template(
        conversation, 
        add_generation_prompt=True, 
        tokenize=True, 
        return_dict=True
    )
    inputs_dict = inputs_dict.to(0, torch.float16)

    output = model.generate(
        **inputs_dict,
        max_new_tokens=50,
        do_sample=False
    )
    
    image = model.generate_image(output[0])
    image.save(output_path)

def generate_lumina(prompt, output_path, model_name="Alpha-VLLM/Lumina-Image-2.0"):
    pipe = Lumina2Pipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=4.0,
        num_inference_steps=50,
        cfg_trunc_ratio=0.25,
        cfg_normalization=True,
        generator=torch.Generator("cpu").manual_seed(42)
    ).images[0]
    
    image.save(output_path)

def generate_main(
    task: str = "dataset/instruction.jsonl",
    model: str = "cogview",
    output_dir: str = "results",
    model_path: str = None,
    vae_path: str = None,
    split: str = "test"
):
    with open(task) as f:
        dataset = [json.loads(line) for line in f]
    validate_message_fmt(dataset)

    os.makedirs(output_dir, exist_ok=True)

    for row in dataset:
        task_id = row["task_id"]
        prompt = row["messages"][0]["content"]
        output_path = os.path.join(output_dir, f"{task_id}.png")

        if model == "cogview":
            generate_cogview(prompt, output_path)
        elif model == "flux":
            generate_flux(prompt, output_path)
        elif model == "janus":
            generate_janus(prompt, output_path)
        elif model == "infinity":
            generate_infinity(prompt, output_path, model_path, vae_path)
        elif model == "sana":
            generate_sana(prompt, output_path)
        elif model == "emu3":
            generate_emu3(prompt, output_path)
        elif model == "lumina":
            generate_lumina(prompt, output_path)
        else:
            raise ValueError(f"Unknown model: {model}")

if __name__ == "__main__":
    from fire import Fire
    Fire(generate_main)