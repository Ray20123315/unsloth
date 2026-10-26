# Custom RVC voice models through the Unsloth Studio API

This change adds account-scoped RVC checkpoint registration to the existing authenticated Studio API. It reuses the existing API server and audio.cpp runtime; Qwen3-TTS keeps its existing workflow.

## Trust policy

Custom uploads are disabled unless the Studio backend process has an exact SHA-256 allowlist configured:

- `UNSLOTH_STUDIO_TRUSTED_RVC_CHECKPOINT_SHA256`: comma-separated approved hashes for `.pth` / `.pt` checkpoint files.
- `UNSLOTH_STUDIO_TRUSTED_RVC_INDEX_SHA256`: comma-separated approved hashes for FAISS `.index` files.

Set these variables in the environment that launches Studio, before starting the existing server. Each value must be a 64-character hex SHA-256 hash. An unconfigured or malformed list disables uploads; an uploaded file whose digest is absent from the list is rejected and its temporary file is removed. The backend stores the approved digest with the private account-scoped record and checks it again before inference, so later file changes or allowlist revocation prevent use.

Only add hashes for files whose provenance you have independently verified and explicitly trust. Computing a hash identifies a file; it does not establish that the file is trustworthy.

Example PowerShell setup (replace the placeholders with approved values and then launch the normal Studio process from this same shell):

~~~powershell
$env:UNSLOTH_STUDIO_TRUSTED_RVC_CHECKPOINT_SHA256 = "<approved-checkpoint-sha256>[,<another-approved-checkpoint-sha256>]"
$env:UNSLOTH_STUDIO_TRUSTED_RVC_INDEX_SHA256 = "<approved-index-sha256>"
~~~

## Register a checkpoint and optional index

Use the same base URL and bearer authentication as the existing Studio API. Upload the raw file body with `application/octet-stream`:

~~~powershell
$base = "http://<UNSLOTH_HOST>:8888"
$headers = @{ Authorization = "Bearer $env:UNSLOTH_API_KEY" }
$checkpoint = Invoke-RestMethod -Method Post -Uri "$base/v1/audio/rvc-models?filename=dafeiyu.pth&name=%E5%A4%A7%E8%82%A5%E9%AD%9A" -Headers $headers -ContentType "application/octet-stream" -InFile "D:\models\dafeiyu.pth"
$checkpoint.id
~~~

A checkpoint can be `.pth` or `.pt` and is capped at 512 MiB. The optional `.index` is also capped at 512 MiB and must have its own approved index hash:

~~~powershell
$index = Invoke-RestMethod -Method Post -Uri "$base/v1/audio/rvc-models/$($checkpoint.id)/index?filename=dafeiyu.index" -Headers $headers -ContentType "application/octet-stream" -InFile "D:\models\dafeiyu.index"
~~~

`GET /v1/audio/rvc-models` lists the current account's registered models. Opaque IDs are account-scoped. API responses never include local file paths. The `/api/inference/audio/rvc-models` prefix is also available to Studio clients. The client selects a model only by ID; it cannot supply a runtime path.

## Checkpoint and index compatibility

audio.cpp currently marks RVC experimental. Its native `.pth` / `.pt` reader accepts a restricted Torch ZIP checkpoint containing RVC `f0`, `config`, and tensor-only `weight` data. The reader uses an allowlisted native pickle-opcode/global parser; it does not call Python `pickle.load` or execute arbitrary checkpoint globals. ZIP64, compressed ZIP entries, unsupported pickle operations/globals, unsupported tensor dtypes/layouts, or incompatible model metadata are rejected. Supported RVC voices must match the audio.cpp v1/v2 tensor layout and 32/40/48 kHz synthesizer rates.

The optional `.index` must be a trained FAISS `IndexIVFFlat` with an `IndexFlatL2` quantizer, L2 metric, and a feature dimension compatible with the selected voice. An arbitrary FAISS index is not guaranteed to load.

The allowlist is still required: a restricted parser reduces code-execution risk but is not a substitute for trusting model provenance, and malformed native parser inputs can still consume resources. The API caps each uploaded file at 512 MiB.

## Convert audio: voice conversion is not an automatic full-song cover

Upload the source track once through the existing `POST /v1/audio/inputs?name=source.wav` endpoint, then submit one conversion run. The API accepts one source input; the backend and audio.cpp handle their configured long-audio split/merge. Custom RVC input is limited to 15 minutes and is also subject to the existing audio-input upload-size limit.

~~~powershell
$source = Invoke-RestMethod -Method Post -Uri "$base/v1/audio/inputs?name=source.wav" -Headers $headers -ContentType "application/octet-stream" -InFile "D:\audio\song.wav"
$body = @{
  workflow = "convert"
  model = "audio-cpp/audio.cpp-gguf/RVC-GGUF"
  inputs = @{ source = @{ input_id = $source.id } }
  convert = @{ mode = "singing"; rvc_model_id = $checkpoint.id }
  options = @{ retrieval_blend = 0.65; semitone_shift = 0 }
} | ConvertTo-Json -Depth 6
$result = Invoke-RestMethod -Method Post -Uri "$base/v1/audio/run" -Headers $headers -ContentType "application/json" -Body $body
$result.clips
~~~

This is **RVC voice conversion**: it transforms the audio samples it receives. If the source is a mixed song, vocals and accompaniment are both in that source and the conversion can affect both. It does not separate vocals, preserve the original backing track, generate a new accompaniment, or create a finished cover mix. A preserve-the-accompaniment cover needs a separate vocal-separation, vocal-conversion, and remix workflow; that workflow is not implemented here. One API request and internal chunking do not change this distinction.

Set `retrieval_blend` to `0` if there is no registered compatible index. Nonzero retrieval requires the model's registered and trusted index. The client must not send `voice_model_path` or `retrieval_index_path`.

## Windows install and rollback

The upstream Windows stable installer (`irm https://unsloth.ai/install.ps1 | iex`) is a packaged install and has no Git checkout to switch. Upstream separately documents a developer install from a Git checkout using `git clone` and `.\install.ps1 --local`; see the [official Studio install instructions](https://github.com/unslothai/unsloth#unsloth-studio-web-ui) and [Windows setup script](https://github.com/unslothai/unsloth/blob/main/studio/setup.ps1). This feature branch is not present in the stable package.

For a packaged install with no checkout, the documented developer-install path is to clone the feature branch and install that source revision:

~~~powershell
git clone --branch feature/dafeiyu-rvc-unsloth https://github.com/Ray20123315/unsloth.git unsloth-dafeiyu-rvc
Set-Location .\unsloth-dafeiyu-rvc
.\install.ps1 --local
unsloth studio -p 8888
~~~

Stop the currently running Studio before launching the selected install. Run one Studio process on the existing API port; this change does not create another API service.

The branch-only helper `studio/scripts/use-dafeiyu-rvc-branch.ps1` cannot be run from `main` before the branch is fetched, because that file does not exist in the `main` checkout. If using an existing source checkout instead, set `$remote` to the remote that points to `Ray20123315/unsloth`; inspect `git remote -v` first if needed. The following bootstrap is usable while still on `main` and does not depend on any feature-only script:

~~~powershell
$remote = "origin"
$dirty = git status --porcelain
if ($LASTEXITCODE -ne 0) { throw "Could not read Git status." }
if ($dirty) { throw "Working tree must be clean; review or save changes first." }
$fetchSpec = "refs/heads/feature/dafeiyu-rvc-unsloth:refs/remotes/$remote/feature/dafeiyu-rvc-unsloth"
git fetch --no-tags $remote $fetchSpec
if ($LASTEXITCODE -ne 0) { throw "Could not fetch the feature branch." }
$backup = "backup/pre-dafeiyu-rvc-" + (Get-Date -Format "yyyyMMdd-HHmmss")
git branch $backup
if ($LASTEXITCODE -ne 0) { throw "Could not create the backup branch." }
git switch --track -c feature/dafeiyu-rvc-unsloth "$remote/feature/dafeiyu-rvc-unsloth"
if ($LASTEXITCODE -ne 0) { throw "Could not switch branches; the backup branch is $backup." }
.\install.ps1 --local
unsloth studio -p 8888
~~~

To roll back a source install, first ensure the checkout is clean, switch to the backup branch printed above, and reinstall that source revision:

~~~powershell
$dirty = git status --porcelain
if ($LASTEXITCODE -ne 0) { throw "Could not read Git status." }
if ($dirty) { throw "Working tree must be clean; review or save changes first." }
git switch backup/pre-dafeiyu-rvc-YYYYMMDD-HHMMSS
if ($LASTEXITCODE -ne 0) { throw "Could not switch to the backup branch." }
.\install.ps1 --local
~~~

The feature-branch `studio/scripts/rollback-dafeiyu-rvc.ps1` is optional and only available after the feature branch has been checked out. The direct rollback above does not depend on it. The scripts only switch source branches; they do not start a second API server or change Windows settings.
