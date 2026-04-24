# Gemma 4 CI workflow staging

GitHub Actions workflow files live under `.github/workflows/` to be
discovered by GitHub. Pushing or modifying anything in that directory
requires a Personal Access Token with the `workflow` scope.

`gemma4-parity.yml` is staged here on the `younghan/gemma4-dev` branch
because the PAT used for routine pushes does not carry that scope. To
land it as an active workflow, do one of:

1. **Regenerate the PAT with `workflow` scope**, then move the file:
   ```bash
   git mv examples/models/gemma4/ci/gemma4-parity.yml \
          .github/workflows/gemma4-parity.yml
   git commit -m "ci: enable gemma4-parity workflow"
   git push origin younghan/gemma4-dev   # works with workflow-scoped PAT
   ```
2. **Upload via GitHub web UI** (drag-and-drop into `.github/workflows/`
   on github.com — the web UI uses your session, not the PAT).
3. **Cherry-pick via a workflow-scoped PAT just for this commit**, then
   revert the PAT setting.

The workflow runs `tests/test_parity.py` (6 layer-by-layer HF tests),
`tests/test_textdec_wrapper.py` (bit-exact text_decoder parity), and
`tests/test_kvcache_decode.py` on PRs touching the Gemma 4 sources or
the shared llama transformer / multimodal runner files. See the file
header for the full path-trigger list.

It needs the `GEMMA4_HF_TOKEN` repository secret to download the
gated `google/gemma-4-E2B-it` checkpoint.
