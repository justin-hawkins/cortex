# Distributed Agentic Task System - Prompt Templates

This directory contains the prompt templates and schemas for the distributed agentic task system.

## Directory Structure
```
/prompts
├── /agents          # System agent prompts
│   ├── coordinator.md
│   ├── decomposer.md
│   ├── complexity_estimator.md
│   ├── qa_reviewer.md
│   └── merge_coordinator.md
├── /workers         # Task execution worker prompts
│   ├── code_general.md
│   ├── code_vision.md
│   ├── code_embedded.md
│   ├── documentation.md
│   └── ui_design.md
├── /schemas         # Data structure definitions
│   ├── task_schema.yaml
│   ├── provenance_schema.yaml
│   └── routing_config.yaml
└── README.md
```

## Template Variables

Prompts use `{variable_name}` syntax for dynamic content injection:

- `{task_id}` - Unique task identifier
- `{task_description}` - What the task should accomplish
- `{inputs}` - Input artifacts and context
- `{acceptance_criteria}` - Success criteria
- `{lightrag_context}` - Relevant project knowledge
- `{constitution}` - Project standards and constraints
- `{model_name}` - Current model being used
- `{model_tier}` - Current tier (tiny/small/large/frontier)
- `{context_window}` - Available context window

## Version Management

Each prompt template should be versioned. When loading templates:
1. Use semantic versioning (e.g., 1.2.3)
2. Track which version produced which outputs in provenance
3. "latest" resolves to highest version number

## Adding New Workers

1. Create new file in `/workers` directory
2. Follow existing template structure
3. Define capabilities and constraints clearly
4. Add domain to routing_config.yaml
5. Update complexity_estimator.md if new routing logic needed