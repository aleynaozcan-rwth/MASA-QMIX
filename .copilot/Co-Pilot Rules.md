# Co-Pilot Rules

Please stop introducing default values, fallback logic, or protective guards that silently replace missing parameters.

I do not want code that:
- silently substitutes a parameter with a default value
- hides errors by using fallback expressions
- interprets "None" as an invitation to generate its own value
- masks configuration mistakes by auto-filling missing arguments
- adds extra layers of indirection (e.g., getattr with default)

When a parameter is missing, I want the code to fail immediately, not continue running with assumed or invented values.

My intention is:
- If a parameter exists → use it.
- If a parameter does not exist → raise an error so I can explicitly define it.
- No automatic assumptions.
- No implicit defaults.
- No silent correction of misconfigurations.

In short:
- Do not generate or infer values on your own.
- Do not hide errors.
- Do not apply fallback logic unless I explicitly ask for it.

Please follow these rules strictly when suggesting code changes.
Thank you.


# Project-Wide Structural Rules

- Always respect the project’s existing architecture, file structure, and logic flow.
- Do not invent new files, folders, parameters, classes, functions, or naming conventions unless explicitly instructed.
- When a modification is required, first search the entire workspace to find where the parameter, variable, or logic is already defined, referenced, or configured.
- Modify the existing definition in its correct location instead of redefining the parameter in another file.
- Never duplicate an existing parameter or configuration in a different file (e.g., do not redefine learning rate, epsilon settings, batch size, n_epochs, n_episodes in runner or environment).
- Always update the canonical source of truth for each variable or configuration.


# No Logic Corruption

- Do not rewrite or restructure the environment, runner, QMIX modules, or replay buffer logic unless specifically instructed.
- Preserve the current pipeline flow:
  Environment → Rollout → Replay Buffer → QMIX → Runner → Training Loop.
- Do not introduce shortcuts, alternative execution paths, or structural rewrites.
- Do not move logic to different files unless explicitly asked.
- Do not introduce “helper wrappers” unless told to.


# Respect Clean Code Principles of This Project

- This project must remain clean, minimal, traceable, and deterministic.
- Every parameter must exist in exactly one well-defined place.
- Parameters must never be scattered across files.
- Avoid adding new names or new abstractions unless absolutely required.
- Always follow existing naming patterns used in this workspace.
- If a concept already exists (e.g., job, JobAgent, WorkCenter, Operator, state_dim, epsilon schedule), do not rename it or reintroduce it.


# Workspace Awareness Rules

- Before writing any code, check whether the project already contains a definition of what you are about to write.
- Use and modify existing files and existing definitions rather than creating new ones.
- Know that this workspace is part of a research thesis. Stability and reproducibility are critical.
- Understand that the project has a defined objective:
  “Develop a clean, reproducible, SimPy-based multi-agent QMIX training environment for flexible job shop scheduling with dynamic arrivals and operator constraints.”
- All code suggestions must support this objective.


# No Silent Behavior Alteration

- Do not modify algorithms (QMIX, Double-Q, epsilon decay, learning rate scheduling) unless explicitly asked.
- Do not change default logic in ways that alter training outcomes or break reproducibility.
- Do not add extra layers, wrappers, abstractions, or protective guards.


# Parameter Handling Rules

- Do not introduce fallback values, default values, or implicit assumptions.
- Do not interpret None as a value that should be replaced.
- A missing parameter must raise an error or require explicit definition.
- All parameters must reflect the single authoritative definition location (arguments.py, configs, or designated source).
- Do not guess or create new parameters.
- Do not override parameters locally inside runner, environment, or agent code. Always update the global config location.


# Memory and Behavior Rules for Copilot

- Treat these rules as permanent constraints for the entire workspace.
- Apply them to all future code suggestions, modifications, refactorings, and completions.
