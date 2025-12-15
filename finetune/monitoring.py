# monitoring.py
import torch
import numpy as np
import wandb
import psutil
import GPUtil
import time
import os
from datetime import datetime
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

class EnhancedMonitoring:
    """All-in-one monitoring solution for LoRA fine-tuning"""
    
    @staticmethod
    def get_callbacks(tokenizer=None, test_prompts=None):
        """Returns all monitoring callbacks"""
        callbacks = [
            EnhancedMonitoring.SystemMonitor(),
            EnhancedMonitoring.VRAMMonitor(),
            EnhancedMonitoring.TrainingProgress(),
        ]
        
        if tokenizer is not None:
            callbacks.append(
                EnhancedMonitoring.ExampleGenerator(tokenizer, test_prompts)
            )
            
        callbacks.append(EnhancedMonitoring.CheckpointSaver())
        return callbacks
    
    @staticmethod
    def init_wandb(project="qwen-finetune", run_name=None, config=None):
        """Initialize wandb with better defaults"""
        if wandb.run is not None:
            wandb.finish()
            
        if run_name is None:
            run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
        wandb.init(
            project=project,
            name=run_name,
            config=config or {},
        )
        return wandb.run
    
    class SystemMonitor(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            
            # GPU
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    logs.update({
                        "gpu/util": gpu.load * 100,
                        "gpu/mem_used_gb": gpu.memoryUsed,
                        "gpu/mem_total_gb": gpu.memoryTotal,
                        "gpu/temp": gpu.temperature,
                    })
            except:
                pass
            
            # CPU/RAM
            logs.update({
                "cpu/util": psutil.cpu_percent(),
                "ram/used_gb": psutil.virtual_memory().used / (1024**3),
                "ram/total_gb": psutil.virtual_memory().total / (1024**3),
            })
            
            # Progress
            if state.epoch is not None:
                logs["progress/epoch"] = state.epoch
            if state.max_steps and state.global_step:
                logs["progress/pct"] = (state.global_step / state.max_steps) * 100
    
    class VRAMMonitor(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                logs = {
                    "vram/allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "vram/reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "vram/max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                }
                if wandb.run:
                    wandb.log(logs, step=state.global_step)
    
    class TrainingProgress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                # Calculate running average loss
                if not hasattr(self, 'loss_history'):
                    self.loss_history = []
                
                self.loss_history.append(logs["loss"])
                if len(self.loss_history) > 10:
                    self.loss_history.pop(0)
                
                logs["loss/sma_10"] = np.mean(self.loss_history)
                
                # Learning rate schedule visualization
                if hasattr(args, 'learning_rate'):
                    logs["lr/current"] = args.learning_rate
    
    class ExampleGenerator(TrainerCallback):
        def __init__(self, tokenizer, test_prompts=None, eval_every=50):
            self.tokenizer = tokenizer
            self.eval_every = eval_every
            self.test_prompts = test_prompts or [
                "Set variable health to 100",
                "Create variable score with value under 200",
            ]
        
        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            if model is None or state.global_step % self.eval_every != 0:
                return
            
            model.eval()
            with torch.no_grad():
                for i, prompt in enumerate(self.test_prompts[:2]):
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    start = time.time()
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    gen_time = time.time() - start
                    
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    if wandb.run:
                        wandb.log({
                            f"eval/prompt_{i}": prompt,
                            f"eval/output_{i}": generated,
                            f"eval/time_{i}": gen_time,
                        }, step=state.global_step)
                    
                    # Console output
                    print(f"\n[Step {state.global_step}]")
                    print(f"Prompt: {prompt}")
                    print(f"Output: {generated[:80]}...")
            
            model.train()
    
    class CheckpointSaver(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = os.path.join(
                args.output_dir, 
                f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            
            # Save metadata
            import json
            metadata = {
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": state.log_history[-1]["loss"] if state.log_history else None,
                "timestamp": datetime.now().isoformat(),
            }
            
            meta_file = os.path.join(checkpoint_dir, "checkpoint_meta.json")
            with open(meta_file, "w") as f:
                json.dump(metadata, f, indent=2)