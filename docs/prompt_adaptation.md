# Prompt Adaptation Rules

This document describes how the orchestration layer adapts subagent prompts using
failure history and context signals.

## Inputs

### Failure history

`failure_history` is a list of records captured after reflection when a subagent
result is incomplete, low quality, or risky. Each entry includes:

- `timestamp`: When the failure was recorded.
- `result_key`: Result category (for example, `iac`, `container`).
- `subagent`: The subagent name.
- `issue`: Evaluation output with `quality`, `completeness`, and `risks`.

### Context signals

`context_signals` is a dictionary of orchestration-level indicators that shape
prompt adaptations. The workflow currently emits:

- `needs_retry`: Boolean indicating a remediation loop is required.
- `risk_tags`: Unique list of risk tags (for example, `execution_failed`).
- `recent_failure_count`: Count of recorded failures.
- `last_failure_subagents`: Up to three recent subagents involved in failures.

The orchestration layer may add additional signals, including:

- `time_sensitive`: Request concise responses.
- `compliance_required`: Require compliance notes.
- `preferred_format`: Preferred output format (for example, `JSON` or `Markdown`).

## Adaptation behavior

When planning subagent invocations, the agent builds a `prompt_profile` with
three sections: `instructions`, `constraints`, and `examples`. The rules below
show how the profile changes:

1. **Baseline guidance**
   - Always include instructions to follow requirements and state assumptions.
   - Always include constraints to avoid speculative changes.

2. **Failure history present**
   - Add instructions to address recent failure modes and the involved
     subagents.
   - Add constraints to explicitly avoid repeating failed approaches.

3. **Risk tags present**
   - Add instructions to mitigate known risk tags.

4. **Retry loop required (`needs_retry`)**
   - Add instructions to provide recovery steps and verification checklist.
   - Add an example that highlights missing-output remediation.

5. **Time sensitive (`time_sensitive`)**
   - Add constraints to keep responses concise and focused on high-impact
     actions.

6. **Compliance required (`compliance_required`)**
   - Add constraints to mention compliance considerations.

7. **Preferred format (`preferred_format`)**
   - Add instructions to format output using the preferred format.

## Example prompt profile

```json
{
  "instructions": [
    "Follow the task requirements precisely and state any assumptions explicitly.",
    "Address prior failure modes from recent subagents: iac-golden-architect.",
    "Mitigate known risk tags: execution_failed.",
    "Provide recovery steps and a verification checklist before final output.",
    "Format the response as: Markdown."
  ],
  "constraints": [
    "Avoid speculative changes outside the provided context and requirements.",
    "Do not repeat previously failed approaches; call out the correction explicitly.",
    "Keep the response concise and prioritize the highest-impact actions."
  ],
  "examples": [
    "Example: If 'missing_output' was flagged, include an explicit Output section."
  ]
}
```
