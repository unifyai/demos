"""Run the experiment. See README.md ("Re-run") for the exact commands.

    ../../bin/with-openrouter-key uv run python run.py --phase main --model anthropic/claude-haiku-5.5
"""

from execution_coordinates.runner import main

if __name__ == "__main__":
    main()
