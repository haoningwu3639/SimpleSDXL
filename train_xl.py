import os
import cv2
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.cuda.amp import autocast
from typing import Optional, Dict
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from utils.util import get_time_string, get_function_args
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline_xl import StableDiffusionXLPipeline
from dataset import SimpleDataset

logger = get_logger(__name__)

class SampleLogger:
    def __init__(
        self,
        logdir: str,
        subdir: str = "sample",
        num_samples_per_prompt: int = 1,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.0,
    ) -> None:
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.num_sample_per_prompt = num_samples_per_prompt
        self.logdir = os.path.join(logdir, subdir)
        os.makedirs(self.logdir)
        
    def log_sample_images(
        self, batch, pipeline: StableDiffusionXLPipeline, device: torch.device, step: int
    ):
        sample_seeds = torch.randint(0, 100000, (self.num_sample_per_prompt,))
        sample_seeds = sorted(sample_seeds.numpy().tolist())
        self.sample_seeds = sample_seeds
        self.prompts = batch["prompt"]
        for idx, prompt in enumerate(tqdm(self.prompts, desc="Generating sample images")):
            image = batch["image"][idx, :, :, :].unsqueeze(0)
            image = image.to(device=device)
            generator = []
            for seed in self.sample_seeds:
                generator_temp = torch.Generator(device=device)
                generator_temp.manual_seed(seed)
                generator.append(generator_temp)    
            
            sequence = pipeline(
                prompt,
                height=image.shape[2],
                width=image.shape[3],
                generator=generator,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                num_images_per_prompt=self.num_sample_per_prompt,
            ).images

            image = (image + 1.) / 2. # for visualization
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            cv2.imwrite(os.path.join(self.logdir, f"{step}_{idx}_{seed}.png"), image[:, :, ::-1] * 255)
            with open(os.path.join(self.logdir, f"{step}_{idx}_{seed}" + '.txt'), 'a') as f:
                f.write(batch['prompt'][idx])
            for i, img in enumerate(sequence):
                img.save(os.path.join(self.logdir, f"{step}_{idx}_{sample_seeds[i]}_output.png"))
            
def train(
    pretrained_model_path: str,
    logdir: str,
    train_steps: int = 5000,
    validation_steps: int = 1000,
    validation_sample_logger: Optional[Dict] = None,
    gradient_accumulation_steps: int = 1, # important hyper-parameter
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    train_batch_size: int = 1,
    val_batch_size: int = 1,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",  # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    lr_warmup_steps: int = 0,
    use_8bit_adam: bool = True,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    checkpointing_steps: int = 10000,
):
    
    args = get_function_args()
    time_string = get_time_string()
    logdir += f"_{time_string}"

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))

    if seed is not None:
        set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae_1_0", low_cpu_mem_usage=False, device_map=None)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    # vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae_1_0")
    # vae = AutoencoderKL.from_config(pretrained_model_path, subfolder="vae_1_0")
    # unet = UNet2DConditionModel.from_config(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
        force_zeros_for_empty_prompt=True,
    )
    pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    trainable_modules = ("attn2")
    for name, module in unet.named_modules():
        if name.endswith(trainable_modules):
            for params in module.parameters():
                params.requires_grad = True
        # for params in module.parameters():
        #     params.requires_grad = True

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    train_dataset = SimpleDataset(root="./")
    val_dataset = SimpleDataset(root="./")
    
    print(train_dataset.__len__())
    print(val_dataset.__len__())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=train_steps * gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    
    # vae.to(accelerator.device)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SimpleSDM")
    step = 0

    if validation_sample_logger is not None and accelerator.is_main_process:
        validation_sample_logger = SampleLogger(**validation_sample_logger, logdir=logdir)

    progress_bar = tqdm(range(step, train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)
    val_data_yielder = make_data_yielder(val_dataloader)

    while step < train_steps:
        batch = next(train_data_yielder)
        
        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        unet.train()
        
        image = batch["image"].to(dtype=weight_dtype)
        prompt = batch["prompt"]
        prompt_ids = tokenizer(prompt, truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
        prompt_ids_2 = tokenizer_2(prompt, truncation=True, padding="max_length", max_length=tokenizer_2.model_max_length, return_tensors="pt").input_ids
        b, c, h, w = image.shape

        latents = vae.encode(image).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        # Sample noise that we'll add
        noise = torch.randn_like(latents) # [-1, 1]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise according to the noise magnitude at each timestep (this is the forward diffusion process)
        noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps)
        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(prompt_ids.to(accelerator.device), output_hidden_states=True, return_dict=False)
        pooled_prompt_embeds = encoder_hidden_states[0]
        encoder_hidden_states = encoder_hidden_states[-1][-2]
        bs_embed, seq_len, _ = encoder_hidden_states.shape
        encoder_hidden_states = encoder_hidden_states.view(bs_embed, seq_len, -1)
        
        encoder_hidden_states_2 = text_encoder_2(prompt_ids_2.to(accelerator.device), output_hidden_states=True, return_dict=False)
        pooled_prompt_embeds_2 = encoder_hidden_states_2[0]
        encoder_hidden_states_2 = encoder_hidden_states_2[-1][-2]
        bs_embed, seq_len, _ = encoder_hidden_states_2.shape
        encoder_hidden_states_2 = encoder_hidden_states_2.view(bs_embed, seq_len, -1)
        
        encoder_hidden_states = torch.concat([encoder_hidden_states, encoder_hidden_states_2], dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds_2.view(bs_embed, -1)
        
        def compute_time_ids(original_size, crops_coords_top_left):
            # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
            target_size = [1024, 1024]
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
            return add_time_ids

        add_time_ids = torch.cat([compute_time_ids(list(s), list(c)) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])])
        
        # refer to https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py for details
        # Predict the noise residual
        unet_added_conditions = {"time_ids": add_time_ids}
        
        unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
        model_pred = unet(
            noisy_latent,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_conditions,
            return_dict=False,
        )[0]
        
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            step += 1
            if accelerator.is_main_process:
                if validation_sample_logger is not None and step % validation_steps == 0:
                    unet.eval()
                    val_batch = next(val_data_yielder)
                    # NOTE: autocast will cause VAE overflow with fp16, will try to fix it later.
                    # with autocast():
                    validation_sample_logger.log_sample_images(
                        batch = val_batch,
                        pipeline=pipeline,
                        device=accelerator.device,
                        step=step,
                    )
                if step % checkpointing_steps == 0:
                    pipeline_save = StableDiffusionXLPipeline(
                        vae=vae,
                        text_encoder=text_encoder,
                        text_encoder_2=text_encoder_2,
                        tokenizer=tokenizer,
                        tokenizer_2=tokenizer_2,
                        unet=accelerator.unwrap_model(unet),
                        scheduler=scheduler,
                    )
                    checkpoint_save_path = os.path.join(logdir, f"checkpoint_{step}")
                    pipeline_save.save_pretrained(checkpoint_save_path)

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=step)
    accelerator.end_training()


if __name__ == "__main__":
    config = "./config.yml"
    train(**OmegaConf.load(config))