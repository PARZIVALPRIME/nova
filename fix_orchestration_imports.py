"""
Fix import paths in orchestration files.
Changes: src.agents.orchestration.guardrails -> src.agents.guardrails
"""

import os
from pathlib import Path

ORCHESTRATION_DIR = Path("D:/trialos/src/agents/orchestration")

# Files to fix
files_to_fix = [
    "agent_registry.py",
    "task_queue.py", 
    "priority_router.py",
    "circuit_breaker.py",
    "fallback_handler.py",
    "health_monitor.py",
    "policy_engine.py",
    "conflict_resolver.py",
    "human_in_loop.py",
    "audit_trail.py",
    "observability.py",
    "orchestration_engine.py"
]

# Wrong import pattern -> correct import pattern
replacements = [
    ("from src.agents.orchestration.guardrails", "from src.agents.guardrails"),
    ("from .guardrails", "from src.agents.guardrails"),
    ("import src.agents.orchestration.guardrails", "import src.agents.guardrails"),
]

fixed_count = 0

for filename in files_to_fix:
    filepath = ORCHESTRATION_DIR / filename
    if filepath.exists():
        content = filepath.read_text(encoding='utf-8')
        original = content
        
        for old, new in replacements:
            content = content.replace(old, new)
            
        if content != original:
            filepath.write_text(content, encoding='utf-8')
            print(f"✅ Fixed: {filename}")
            fixed_count += 1
        else:
            print(f"⏭️  No changes needed: {filename}")
    else:
        print(f"❌ File not found: {filename}")

print(f"\n✅ Fixed {fixed_count} files")