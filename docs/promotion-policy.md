# Model Promotion Policy (Example)

This is a simple, teachable policy for demo purposes.

## Promotion stages (common meanings)
- **None / Unassigned**: Newly registered versions before any promotion.
- **Staging**: Candidate for production; validated or approved for limited testing.
- **Production**: The version the live system should serve.
- **Archived**: Retired versions kept for audit and rollback history.

Teams may also add **custom tags or aliases** (e.g., `candidate`, `champion`, `challenger`) as workflow labels.

## How a system uses the policy
In a real system, the policy is enforced by a combination of automation and human approval:

- **Pipeline log step** logs a run and registers a model version after train/eval artifacts are ready.
- **Validation checks** (CI or scheduled jobs) evaluate metrics and artifacts.
- **Approval step** records a decision to promote or reject.
- **Deployment system** pulls the model version from the Registry based on stage or alias.

In practice, “promotion” is how the system knows what to serve in an environment:
- `Staging` might map to a canary or QA deployment.
- `Production` is the default model used by the live service.

## Staging criteria
- `test_roc_auc` above a defined threshold (e.g., 0.97)
- No obvious regressions in `test_precision` / `test_recall`
- Artifacts (confusion matrix, ROC/PR curves) look reasonable

## Promotion steps
1) Compare the new run to the current Staging run in MLflow.
2) If metrics improve (or stay within acceptable bounds), promote:
   - `Staging` → `Production` (optional in dev)
3) Record the reason for promotion in a run tag or in a simple changelog.

## Example automation flow
1) **Train** → register model version.
2) **Evaluate** → compute metrics vs. thresholds.
3) **Gate**:
   - If all checks pass, mark as `Staging`.
   - If manual approval is required, wait for approval.
4) **Deploy**:
   - Serving system queries Registry for `Production` stage.
   - Deployment updates to that version automatically.

## Operational considerations
- **Auditability**: log who/why promoted a model (tag or changelog).
- **Safety**: use canary or shadow deployments for Staging.
- **Monitoring**: watch for data drift or metric drops post‑deploy.
- **Rollback**: reverting stage is the fastest way to undo a bad release.

## Rollback
If issues appear, revert the stage to a previous version in the Registry.
