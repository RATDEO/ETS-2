# Tiny-Qwen CPU refinement smoke test

This run checks the refinement interface only. It is not a substitute for the owner's Qwen 27B server and is not a statistically powered performance claim.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Frozen historical origins: **8**
- Parse success: **8/8**
- Fallbacks: **0**
- Base path MSE: **25.280014**
- Refined path MSE: **25.296493**
- Provisional change: **-0.065%**
- Median CPU generation latency: **4.98s/case**
- Mean absolute applied adjustment: **0.011%**

Any final LLM-refinement conclusion must be rerun on newly frozen windows using the exact Qwen 27B model served by the owner's GPU server.
