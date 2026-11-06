# Evaluated integration snapshots

These files preserve the experiment integration used for the reported pilot. They are not standalone benchmark installers. They import pinned upstream code and expect its simulator, checkpoints, metadata and declared case/memory assets. Historical freeze receipts may refer to an original workspace; do not bypass those guards or manufacture passing receipts.

Use the root harness module as the reusable API. For a new evaluation, prepare the upstream runtime, declare fresh output/case manifests, verify native-path parity and horizon accounting, then freeze sources before evaluation. Do not label a new environment as an exact historical reproduction merely because these scripts import successfully.
