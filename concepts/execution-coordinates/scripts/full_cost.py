"""What the full design would cost on the mid and top tiers (see analysis.full_design_cost).

    uv run python scripts/full_cost.py
"""

from execution_coordinates.analysis import full_design_cost

print("| model | requests run | cost of those | full design requests | full design, estimated | the rest |")
print("|---|---|---|---|---|---|")
for model, d in full_design_cost().items():
    print(f"| {model} | {d['requests_run']:,} | ${d['cost_run']:.2f} | {d['full_requests']:,} | "
          f"${d['full_cost_estimate']:.0f} | ${d['rest_cost_estimate']:.0f} |")
