# Off-Machine Recovery Archive

MMM-recovery-20261010-081618.zip preserves the dated recovery package: project.zip (source, Git history, saved data/results/posterior draws), raw VS Code workspace chat records, environment inventory, and the original SHA-256 manifest.

Keep this repository private. Raw chat records can contain sensitive prompts, code, local paths, and tool output. They are a snapshot of available workspace history, not a supported Chat: Export Chat file or a guarantee of complete history across every workspace/session.

To recover, extract the outer archive, verify its contents against checksums.csv, and extract project.zip into a separate project folder. Read CHECKPOINT.md before running anything. Recreate the Python environment using the recorded versions; do not rerun MCMC just to resume. Preserve saved posterior draws for exact fitted results.

The project ZIP contains the earlier checkpoint's .git directory. Extract it separately rather than over an existing checkout. The outer archive is immutable historical backup; newer conversations and results require a new dated snapshot.