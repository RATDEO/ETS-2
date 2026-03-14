# UK ETS 35B Tool Follow-Up Sweep Plan

Generated: 2026-03-14T03:01:54.986471

## Planned Candidates

| Candidate | Literature | Aim | Expected range | Change |
|---|---|---|---|---|
| optional_tools_default | [Toolformer](https://arxiv.org/abs/2302.04761) | Test whether 35B is being over-regularized by forced tool calls on easy windows. | 21.42-21.56 | Keep the current default stack but make numeric and verifier tool use optional instead of required. |
| numeric_only_default | [Program of Thoughts](https://arxiv.org/abs/2211.12588) | Test whether exact arithmetic helps more than post-hoc clipping on 35B. | 21.38-21.54 | Remove the verifier and keep only the numeric tool under the current least-to-most stack. |
| verifier_only_default | [Chain-of-Verification](https://arxiv.org/abs/2309.11495) | Test whether 35B already has enough arithmetic and mainly needs bounded final corrections. | 21.43-21.57 | Remove the numeric tool and keep only the deterministic verifier under the current least-to-most stack. |
| react_evidence_optional_tools | [ReAct](https://arxiv.org/abs/2210.03629) | Encourage evidence-driven tool use instead of mandatory tool loops. | 21.36-21.55 | Switch the reflection style to evidence-seeking ReAct and let the model decide when to call numeric or verifier tools. |
| plan_and_solve_optional_tools | [Plan-and-Solve](https://arxiv.org/abs/2305.04091) | Reduce missing-step errors without forcing every sample through the same tool sequence. | 21.38-21.56 | Use plan-and-solve reflection while keeping numeric and verifier tools optional. |
| skill_tag_recent_high_error_tools | [Skill-KNN](https://arxiv.org/abs/2305.14210) | Improve demonstration relevance without changing the tool contract. | 21.36-21.53 | Switch example selection to skill-tag-matched recent hard cases while keeping the current tool stack forced. |
| balanced_long_horizon_tools | [Active Example Selection](https://arxiv.org/abs/2211.04486) | Avoid overfitting the demonstration bank to one long-horizon failure mode. | 21.39-21.56 | Balance the teaching pool across h20-heavy and h30-heavy failures under the current tool stack. |
| counterexample_recent_high_error_tools | [Chain of Hindsight](https://arxiv.org/abs/2302.02676) | Teach the model both when to correct and when to leave the base path alone. | 21.37-21.55 | Add one low-error freeze counterexample to the recent high-error teaching set while keeping the current tool stack. |
| step_back_optional_tools_k6 | [Step-Back Prompting](https://arxiv.org/abs/2310.06117) | Recover the old 35B regime-level strength without forcing the clipping behavior that hurt the hybrid. | 21.30-21.50 | Restore step-back with 6 examples but leave numeric and verifier tools available rather than required. |
| step_back_numeric_optional_k6 | [Step-Back Prompting + Program of Thoughts](https://arxiv.org/abs/2310.06117) | See whether exact arithmetic complements regime-level reasoning while the verifier is the component causing over-regularization. | 21.28-21.48 | Use step-back with 6 examples, enable numeric analysis as-needed, and remove the verifier entirely. |
