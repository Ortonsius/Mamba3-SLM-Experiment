# Mamba3-SLM-Experiment
Pretraining code to train SLM/LLM using Mamba3 as backbone architecture.

## Sources
- Paper: https://arxiv.org/pdf/2603.15569
- Original code: https://github.com/state-spaces/mamba
- Mamba3 original models: https://huggingface.co/state-spaces

## Important Details:
- Use pre-gate default/grouped RMS. Applying RMS norm after skip connection helps model for long conversation.
- Use hybrid model (by inserting attention block after N Mamba3 layers) if you want to build LLM.
- The hybrid model proposed by the author use NoPE (no positional encoding) at attention layer to improve retrieval of long context.
- Increasing d_state causing large memory usage on training. This might happen because the einsum operation. Feel free to optimize the code if you want better efficiency.
- The original code use heavy tail activation after A projection. see: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py#L194

## Notes:
- This code is experimental only and not proven yet to build a production ready LLM.
- If you find this code not correct yet, feel free to fix and optimize it.
