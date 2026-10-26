# Custom RVC voice models through the Unsloth Studio API

This change adds custom RVC checkpoint registration to the existing authenticated Studio API. It keeps the existing server and Qwen3-TTS routes. The client selects a registered model by opaque ID; filesystem paths are created and resolved only inside the server.

## Register the checkpoint and optional index

Use the same base URL and bearer authentication as the existing Studio API. Upload the raw file body with application/octet-stream:

~~~powershell
$base = "http://<UNSLOTH_HOST>:8888"
$headers = @{ Authorization = "Bearer $env:UNSLOTH_API_KEY" }
$checkpoint = Invoke-RestMethod -Method Post -Uri "$base/v1/audio/rvc-models?filename=dafeiyu.pth&name=%E5%A4%A7%E8%82%A5%E9%AD%9A" -Headers $headers -ContentType "application/octet-stream" -InFile "D:\models\dafeiyu.pth"
$checkpoint.id
~~~

A checkpoint may be .pth or .pt and is capped at 512 MiB. The optional FAISS .index is capped at 512 MiB:

~~~powershell
$index = Invoke-RestMethod -Method Post -Uri "$base/v1/audio/rvc-models/$($checkpoint.id)/index?filename=dafeiyu.index" -Headers $headers -ContentType "application/octet-stream" -InFile "D:\models\dafeiyu.index"
~~~

GET /v1/audio/rvc-models lists the current account's registered models. IDs and metadata are account-scoped. Responses contain no local file paths. The /api/inference/audio/rvc-models prefix is also available to Studio clients.

## Convert a complete song

First upload the original track once through the existing POST /v1/audio/inputs?name=source.wav endpoint. Then submit one run; the backend prepares and converts the complete source, and audio.cpp performs its configured long-audio split/merge internally.

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

Set retrieval_blend to 0 if no index is registered. Nonzero retrieval requires the model's registered index. Custom RVC inputs may be up to 15 minutes; other conversion families keep their existing 5-minute limit. The client must not send voice_model_path or retrieval_index_path.

The runtime still needs the existing audio.cpp RVC package (RVC-GGUF) available in Studio. Qwen3-TTS remains on its current workflow. RVC is marked experimental by the upstream audio.cpp model specification.

## Apply the branch on Windows

From the existing Unsloth source checkout, ensure its working tree is clean. If the fork remote is not named origin, pass its remote name:

~~~powershell
.\studio\scripts\use-dafeiyu-rvc-branch.ps1 -Remote origin
~~~

The script fetches the feature branch, creates a timestamped backup branch at the current commit, and switches to the feature branch. It refuses a dirty checkout or an already-existing local target branch. Review the changes, then restart Unsloth Studio using the normal installation workflow. The script does not run a service, change Windows settings, or install packages.

## Roll back

Use the backup branch name printed by the apply script:

~~~powershell
.\studio\scripts\rollback-dafeiyu-rvc.ps1 -BackupBranch backup/pre-dafeiyu-rvc-YYYYMMDD-HHMMSS
~~~

Rollback also requires a clean checkout and only switches back to that backup branch.
