param(
    [string]$Phase = "all",
    [int]$GpuCount = 0,                       # 0 = interactive prompt (preskoci se pro 'finalize')
    [ValidateSet("b200", "rtxpro6000", "a100")]
    [string]$GpuType = "",                    # "" = prompt (s -GpuCount: nvidia-smi, jinak b200)
    [string]$TaxonomySamples = "",            # auto | cislo; "" = prompt (Enter = .env / auto)
    [ValidateSet("docker", "native")]
    [string]$Runtime = "docker"               # docker (default) / native (host Python proti bezicimu LLM)
)

# Pouziti:
#   .\scripts\run_production.ps1 all              cela pipeline (0a -> finalize)
#   .\scripts\run_production.ps1 taxonomy         jedna faze
#   .\scripts\run_production.ps1 from-extract     od extract dal
#
# Faze: sample | taxonomy | extract | critic | revise | finalize | all
#       from-sample | from-taxonomy | from-extract | from-critic | from-revise
#
# Kdyz beh zacina na/za `extract`, faze 0 se NESPOUSTI - extract si nacte
# existujici checkpoints\entity_taxonomy.json. Vlastni rezim si vynutis pres
# $env:EXTRACT_EXTRA_ARGS="--taxonomy-mode auto".
#
# Dotaz na soubeznost vypnes $env:THROUGHPUT_PROMPT="0".
#
# Pouziti (native runtime):
#   # 1) Sam si spust LLM server, napr.:
#   #    python -m sglang.launch_server --model-path RedHatAI/Qwen3.5-122B-A10B-NVFP4 ...
#   # 2) Spust skript v native modu:
#   .\scripts\run_production.ps1 all -Runtime native

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "..")

$ExtractorBackends = if ($env:EXTRACTOR_BACKENDS) { $env:EXTRACTOR_BACKENDS -split ' ' } else { @("sglang", "vllm") }
$CriticBackends    = if ($env:CRITIC_BACKENDS)    { $env:CRITIC_BACKENDS    -split ' ' } else { @("sglang", "vllm") }
$ReviseBackends    = if ($env:REVISE_BACKENDS)    { $env:REVISE_BACKENDS    -split ' ' } else { @("sglang", "vllm") }

function Write-Log  { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Err  { param($m) Write-Host "`n!! $m" -ForegroundColor Red }
function Write-Warn { param($m) Write-Host "`n!! $m" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# Preflight checks — chyti bezne konfiguracni problemy PRED docker compose
# ---------------------------------------------------------------------------
function Get-DetectedGpuCount {
    # PS 5.1: redirect stderr nativniho prikazu (2>$null / 2>&1) bali radky
    # do ErrorRecordu; s globalnim $ErrorActionPreference=Stop by PRVNI progress
    # hlaska dockeru (jde na stderr) shodila skript. Lokalni Continue plati
    # jen pro tuto funkci.
    $ErrorActionPreference = "Continue"
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return 0
    }
    $out = & nvidia-smi --list-gpus 2>$null | Measure-Object | Select-Object -ExpandProperty Count
    if ($out -eq $null) { return 0 }
    return [int]$out
}

function Test-QuantizationPreflight {
    $model = if ($env:EXTRACTOR_MODEL) { $env:EXTRACTOR_MODEL } else { "RedHatAI/Qwen3.5-122B-A10B-NVFP4" }
    $quant = $env:EXTRACTOR_QUANTIZATION
    if ($quant -and $model -match "(NVFP4|FP4|nvfp4|fp4)") {
        Write-Err "Quantization mismatch detected:"
        Write-Err "  EXTRACTOR_MODEL=$model (NVFP4 model)"
        Write-Err "  EXTRACTOR_QUANTIZATION=$quant (manualni override)"
        Write-Err ""
        Write-Err "NVFP4 modely maji quantization v config.json (compressed-tensors)"
        Write-Err "- SGLang/vLLM si ji auto-detect. Rucni override = ValueError."
        Write-Err ""
        Write-Err "Fix: zakomentuj EXTRACTOR_QUANTIZATION v .env"
        throw "Quantization mismatch (EXTRACTOR)"
    }
    $cmodel = if ($env:CRITIC_MODEL) { $env:CRITIC_MODEL } else { "nvidia/Gemma-4-31B-IT-NVFP4" }
    $cquant = $env:CRITIC_QUANTIZATION
    if ($cquant -and $cmodel -match "(NVFP4|FP4|nvfp4|fp4)") {
        Write-Err "Quantization mismatch detected:"
        Write-Err "  CRITIC_MODEL=$cmodel (NVFP4 model)"
        Write-Err "  CRITIC_QUANTIZATION=$cquant (manualni override)"
        Write-Err "Fix: zakomentuj CRITIC_QUANTIZATION v .env"
        throw "Quantization mismatch (CRITIC)"
    }
}

function Test-HfTokenPreflight {
    $t = $env:HF_TOKEN
    if (-not $t -or $t -eq "hf_TADY_DOPLN_TOKEN") {
        Write-Warn "HF_TOKEN neni nastaveny nebo ma placeholder hodnotu."
        Write-Warn "Gated modely (Gemma, restricted Llama) NEPUJDOU stahnout."
        Write-Warn "Doplnte HF_TOKEN v .env."
    }
}

# ---------------------------------------------------------------------------
# Interactive GPU prompt (s GPU count validation)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# .env auto-config
# ---------------------------------------------------------------------------
function Update-EnvVar {
    param([string]$File, [string]$Key, [string]$Value)
    if (-not (Test-Path $File)) { return }
    $lines = Get-Content $File
    $found = $false
    $newLines = foreach ($line in $lines) {
        if ($line -match "^[ \t]*${Key}=") {
            $found = $true
            "${Key}=${Value}"
        } else {
            $line
        }
    }
    if (-not $found) {
        $newLines = @($newLines) + "${Key}=${Value}"
    }
    Set-Content -Path $File -Value $newLines -Encoding utf8
}

# ---------------------------------------------------------------------------
# HW presety per typ GPU (modely, kvantizace, backendy, VRAM, kontext)
# ---------------------------------------------------------------------------
# Blackwell (b200 / rtxpro6000) -> NVFP4 modely, SGLang primarne.
# Ampere (a100 40 GB)           -> SGLang jako na Blackwellu, ale INT4 misto
#   NVFP4/FP8: NVFP4 chce Blackwell a FP8 u MoE na SM80 nejde v zadnem enginu
#   (chybi fp8e4nv; sglang#12887, #7715, vllm#18008, vllm#54318). INT4 W4A16
#   pres Marlin MoE ale SGLang na Ampere umi (overeno na SM80/SM86):
#   extraktor Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 (~65 GB vah, TP=4 -> ~17 GB/GPU),
#   critic je dense a jede oficialni QAT google/gemma-4-31B-it-qat-w4a16-ct
#   (~20 GB vah) -> TP=2. Backendy zustavaji "sglang vllm" (vLLM = zaloha).
#   Pokud SGLang Marlin MoE spadne na dtype aktivaci, nastav
#   EXTRACTOR_DTYPE=float16.
#
# Modely/parametry se prepisou JEN kdyz v .env drzi nekterou ze spravovanych
# hodnot (nebo nic) - vlastni model uzivatele zustane.
$script:ManagedExtractorModels = @(
    "RedHatAI/Qwen3.5-122B-A10B-NVFP4",
    "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
    "QuantTrio/Qwen3.5-122B-A10B-AWQ",
    "Intel/Qwen3.5-122B-A10B-int4-AutoRound",
    "RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic",
    "Qwen/Qwen3.5-122B-A10B-FP8",
    "Qwen/Qwen3.5-122B-A10B"
)
$script:ManagedCriticModels = @(
    "nvidia/Gemma-4-31B-IT-NVFP4",
    "google/gemma-4-31B-it-qat-w4a16-ct",
    "RedHatAI/gemma-4-31B-it-FP8-block",
    "google/gemma-4-31B-it"
)
# Hodnoty *_BATCH_SIZE, ktere nastavuji samotne presety - jen ty se prepisuji
# pri zmene HW. Vlastni cislo (z auto_tune nebo z dotazu pri startu) zustava.
$script:ManagedExtractorBatch = @("8", "32")
$script:ManagedCriticBatch = @("16", "64")

function Read-EnvVarValue {
    param([string]$File, [string]$Key)
    if (-not (Test-Path $File)) { return "" }
    $val = ""
    foreach ($line in (Get-Content $File)) {
        if ($line -match "^[ 	TABESC]*${Key}=(.*)$") { $val = $Matches[1].Trim() }
    }
    return $val
}

# Zapise hodnotu jen pokud je aktualni prazdna nebo patri do spravovaneho seznamu.
function Update-EnvVarIfManaged {
    param([string]$File, [string]$Key, [string]$Value, [string[]]$Managed)
    $cur = Read-EnvVarValue -File $File -Key $Key
    if (-not $cur) {
        Update-EnvVar -File $File -Key $Key -Value $Value
        return
    }
    if ($Managed -contains $cur) {
        if ($cur -ne $Value) { Update-EnvVar -File $File -Key $Key -Value $Value }
        return
    }
    Write-Warn "$Key='$cur' je vlastni hodnota - nechavam ji (HW preset by nastavil '$Value')."
}

# Zvedne hodnotu na minimum (nizsi prepise, vyssi uzivatelovu necha).
function Update-EnvVarMinimum {
    param([string]$File, [string]$Key, [int]$Min, [string]$Reason)
    $cur = Read-EnvVarValue -File $File -Key $Key
    $curInt = 0
    if ($cur -match '^\d+') { $curInt = [int]($cur -replace '\\.*$', '') }
    if ($curInt -lt $Min) {
        Update-EnvVar -File $File -Key $Key -Value "$Min"
        Write-Log "Auto-config: $Key=$Min ($Reason)"
    }
}

function Set-GpuPreset {
    param([string]$File, [string]$Type, [int]$Count)
    if ($Type -eq "a100") {
        # INT4 W4A16 - to SGLang i vLLM na Ampere umi pres Marlin (FP8 ne).
        Update-EnvVarIfManaged -File $File -Key "EXTRACTOR_MODEL" `
            -Value "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" -Managed $script:ManagedExtractorModels
        Update-EnvVarIfManaged -File $File -Key "CRITIC_MODEL" `
            -Value "google/gemma-4-31B-it" -Managed $script:ManagedCriticModels
        # Qwen3.5 MoE INT4 v SGLangu jede Triton w4a16 cestou (moe_wna16),
        # NE Marlinem: Marlin MoE si pri prepacku vyrobi fp16 scales, takze
        # v bf16 spadne assert v moe_wna16_marlin_gemm a ve fp16 zas GDN conv
        # cache, ktera zustava bf16 (sglang#30178). moe_wna16 = bf16 vsude.
        Update-EnvVar -File $File -Key "EXTRACTOR_QUANTIZATION" -Value "moe_wna16"
        Update-EnvVar -File $File -Key "CRITIC_QUANTIZATION" -Value ""
        # Dtype nechavame na auto-detectu (bf16) - fp16 uz neni potreba
        Update-EnvVar -File $File -Key "EXTRACTOR_DTYPE" -Value ""
        Update-EnvVar -File $File -Key "CRITIC_DTYPE" -Value ""
        # GDN conv cache MUSI mit stejny dtype jako model - zbytkovy float16
        # z predchoziho pokusu shodi Triton kernel causal_conv1d.
        Update-EnvVar -File $File -Key "SGLANG_MAMBA_CONV_DTYPE" -Value ""
        # Vsechny faze na SGLangu, vLLM jen jako zaloha.
        Update-EnvVar -File $File -Key "EXTRACTOR_BACKENDS" -Value "sglang vllm"
        Update-EnvVar -File $File -Key "CRITIC_BACKENDS" -Value "sglang vllm"
        Update-EnvVar -File $File -Key "REVISE_BACKENDS" -Value "sglang vllm"
        # Critic bf16 Gemma 4 31B ~62 GB vah -> TP=4 (~15,5 GB/GPU).
        # Kvantizovane varianty by se vlezly na 2 karty, ale SGLang je na
        # Ampere nenalozi (QAT checkpoint ma kvantizovany jen jazykovy model
        # a SGLang pro vision tower vyrobi params, ktere v nem nejsou).
        # Faze jedou po sobe, takze 4 karty pro critic nic neblokuji.
        Update-EnvVar -File $File -Key "CRITIC_TP_SIZE" -Value "4"
        # Triton moe_wna16 + prefill CUDA grafy se na 40GB karte pri 0.92
        # nevejdou (dokumentovany priklad jede 0.8-0.85).
        Update-EnvVar -File $File -Key "EXTRACTOR_MEM_FRACTION" -Value "0.85"
        Update-EnvVar -File $File -Key "CRITIC_MEM_FRACTION" -Value "0.88"
        # Extraktor 64k: prompt faze 0 mel pres 31k tokenu a do 32k se
        # s odpovedi nevejde. Max kontext nealokuje pamet (KV pool urcuje
        # *_MEM_FRACTION). Critic ma per-item prompty -> 32k staci.
        Update-EnvVar -File $File -Key "EXTRACTOR_CONTEXT_LEN" -Value "65536"
        Update-EnvVar -File $File -Key "CRITIC_CONTEXT_LEN" -Value "32768"
        Update-EnvVar -File $File -Key "EXTRACTOR_MAX_TOKENS" -Value "16384"
        Update-EnvVar -File $File -Key "CRITIC_MAX_TOKENS" -Value "24576"
        Update-EnvVar -File $File -Key "EXTRACTOR_NER_MAX_TOKENS" -Value "8192"
        # Server-side davky (max-running-requests) - nealokuji pamet, KV pool
        # je dan *_MEM_FRACTION. Mala davka = dequant vah se rozpusti do mala
        # tokenu -> karty jen na ~40 % TDP. Optimum zmeri auto_tune_concurrency.
        # *_CONCURRENCY zamerne NEPREPISUJEME (drzi si .env).
        Update-EnvVarIfManaged -File $File -Key "EXTRACTOR_BATCH_SIZE" `
            -Value "32" -Managed $script:ManagedExtractorBatch
        Update-EnvVarIfManaged -File $File -Key "CRITIC_BATCH_SIZE" `
            -Value "64" -Managed $script:ManagedCriticBatch
        # Marlin dequant do BF16 + male davky = pomalejsi per-request
        Update-EnvVarMinimum -File $File -Key "LLM_TIMEOUT_S" -Min 7200 `
            -Reason "A100: Marlin dequant, mensi davky"
        Write-Log ("Auto-config ${File}: A100 preset - extraktor GPTQ-Int4 " +
                   "(moe_wna16, TP=$Count, kontext 65536), critic bf16 Gemma " +
                   "(TP=4, kontext 32768), backend sglang+vllm")
        return
    }
    # Blackwell -> NVFP4 (plny NVFP4 stack, SGLang primarne)
    Update-EnvVarIfManaged -File $File -Key "EXTRACTOR_MODEL" `
        -Value "RedHatAI/Qwen3.5-122B-A10B-NVFP4" -Managed $script:ManagedExtractorModels
    Update-EnvVarIfManaged -File $File -Key "CRITIC_MODEL" `
        -Value "nvidia/Gemma-4-31B-IT-NVFP4" -Managed $script:ManagedCriticModels
    Update-EnvVar -File $File -Key "EXTRACTOR_QUANTIZATION" -Value ""
    Update-EnvVar -File $File -Key "CRITIC_QUANTIZATION" -Value ""
    # NVFP4 jede v bf16 -> dtype nechavame na auto-detectu
    Update-EnvVar -File $File -Key "EXTRACTOR_DTYPE" -Value ""
    Update-EnvVar -File $File -Key "CRITIC_DTYPE" -Value ""
    # conv cache v sync s modelem (viz A100 vyse)
    Update-EnvVar -File $File -Key "SGLANG_MAMBA_CONV_DTYPE" -Value ""
    Update-EnvVar -File $File -Key "EXTRACTOR_BACKENDS" -Value "sglang vllm"
    Update-EnvVar -File $File -Key "CRITIC_BACKENDS" -Value "sglang vllm"
    Update-EnvVar -File $File -Key "REVISE_BACKENDS" -Value "sglang vllm"
    Update-EnvVar -File $File -Key "CRITIC_TP_SIZE" -Value "1"
    Update-EnvVar -File $File -Key "EXTRACTOR_MEM_FRACTION" -Value "0.85"
    Update-EnvVar -File $File -Key "CRITIC_MEM_FRACTION" -Value "0.85"
    Update-EnvVar -File $File -Key "EXTRACTOR_CONTEXT_LEN" -Value "65536"
    Update-EnvVar -File $File -Key "CRITIC_CONTEXT_LEN" -Value "65536"
    Update-EnvVar -File $File -Key "EXTRACTOR_MAX_TOKENS" -Value "24576"
    Update-EnvVar -File $File -Key "CRITIC_MAX_TOKENS" -Value "49152"
    Update-EnvVar -File $File -Key "EXTRACTOR_NER_MAX_TOKENS" -Value "8192"
    # *_CONCURRENCY nechavame na .env / auto_tune_concurrency.sh
    Update-EnvVarIfManaged -File $File -Key "EXTRACTOR_BATCH_SIZE" `
        -Value "32" -Managed $script:ManagedExtractorBatch
    Update-EnvVarIfManaged -File $File -Key "CRITIC_BATCH_SIZE" `
        -Value "64" -Managed $script:ManagedCriticBatch
    if ($Type -eq "rtxpro6000") {
        # GDDR7 + TP pres PCIe -> delsi worst-case request
        Update-EnvVarMinimum -File $File -Key "LLM_TIMEOUT_S" -Min 3600 -Reason "RTX PRO 6000"
    }
}

function Set-EnvForGpuCount {
    param([int]$Count, [string]$Type = "b200")
    $envFile = if ($env:ENV_FILE) { $env:ENV_FILE } else { ".env" }
    if (-not (Test-Path $envFile)) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" $envFile
            Write-Log "Vytvoreno $envFile z .env.example (auto-config)"
        } else {
            Write-Warn "Neexistuje $envFile ani .env.example — auto-config preskocen."
            return
        }
    }
    Update-EnvVar -File $envFile -Key "GPU_TYPE" -Value $Type
    Update-EnvVar -File $envFile -Key "GPU_COUNT" -Value "$Count"
    Update-EnvVar -File $envFile -Key "EXTRACTOR_TP_SIZE" -Value "$Count"
    # HW preset: modely + kvantizace + backendy + VRAM + kontext + davky
    # (NVFP4 na Blackwellu, INT4/FP8 + vLLM na Ampere - viz Set-GpuPreset).
    Set-GpuPreset -File $envFile -Type $Type -Count $Count
    $criticTp = Read-EnvVarValue -File $envFile -Key "CRITIC_TP_SIZE"
    if (-not $criticTp) { $criticTp = "1" }
    Write-Log "Auto-config ${envFile}: GPU_TYPE=$Type GPU_COUNT=$Count EXTRACTOR_TP_SIZE=$Count CRITIC_TP_SIZE=$criticTp"
}

# ---------------------------------------------------------------------------
# Typ GPU (GpuType):
#   b200        NVIDIA B200, 180 GB HBM3e — Qwen3.5-122B-NVFP4 (~80 GB vah)
#               se vejde na 1 GPU -> 1, 2, 4 nebo 8 GPU
#   rtxpro6000  NVIDIA RTX PRO 6000 Blackwell, 96 GB GDDR7 — Qwen se na jednu
#               kartu nevejde (vahy + KV cache) -> extractor TP >= 2 -> 2, 4, 8
#   a100        NVIDIA A100 40 GB (Ampere) - NVFP4/FP8 tam nejde, jede INT4
#               W4A16 pres Marlin (SGLang i vLLM) -> 4 nebo 8 GPU
# Pocty jen power-of-2 (TP=3/5/6/7 SGLang/vLLM pro MoE nepodporuji).
# ---------------------------------------------------------------------------
function Get-GpuTypeLabel {
    param([string]$Type)
    if ($Type -eq "rtxpro6000") { return "RTX PRO 6000" }
    if ($Type -eq "a100") { return "A100 40GB" }
    if ($Type -eq "b200") { return "B200" }
    return $Type
}

function Get-GpuTypeCounts {
    param([string]$Type)
    if ($Type -eq "rtxpro6000") { return @(2, 4, 8) }
    if ($Type -eq "a100") { return @(4, 8) }
    return @(1, 2, 4, 8)
}

# Typ GPU podle nazvu prvni karty v nvidia-smi; "" = nerozpoznano.
function Get-DetectedGpuType {
    # PS 5.1: redirect stderr nativniho prikazu (2>$null / 2>&1) bali radky
    # do ErrorRecordu; s globalnim $ErrorActionPreference=Stop by PRVNI progress
    # hlaska dockeru (jde na stderr) shodila skript. Lokalni Continue plati
    # jen pro tuto funkci.
    $ErrorActionPreference = "Continue"
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return "" }
    $name = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
    if ($name -match "B200") { return "b200" }
    if ($name -match "RTX PRO 6000") { return "rtxpro6000" }
    if ($name -match "A100") { return "a100" }
    return ""
}

function Read-GpuType {
    param([string]$Detected)
    $default = ""
    if ($Detected -eq "b200") { $default = "1" }
    if ($Detected -eq "rtxpro6000") { $default = "2" }
    if ($Detected -eq "a100") { $default = "3" }
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════"
    Write-Host "  Jaky typ GPU mas k dispozici pro pipeline?"
    Write-Host "    1) NVIDIA B200          (180 GB) - NVFP4, Qwen 122B na 1 GPU"
    Write-Host "    2) NVIDIA RTX PRO 6000  (96 GB)  - NVFP4, Qwen 122B min. 2 GPU"
    Write-Host "    3) NVIDIA A100          (40 GB)  - INT4 (W4A16), Qwen 122B min. 4 GPU"
    if ($Detected) {
        Write-Host "  -> Detekovano: $(Get-GpuTypeLabel $Detected)"
    }
    Write-Host "════════════════════════════════════════════════════════"
    $prompt = if ($default) { "Vyber typ GPU [1/2/3] (Enter = $default)" } else { "Vyber typ GPU [1/2/3]" }
    while ($true) {
        $t = Read-Host $prompt
        if (-not $t) { $t = $default }
        if ($t -eq "1") { return "b200" }
        if ($t -eq "2") { return "rtxpro6000" }
        if ($t -eq "3") { return "a100" }
        Write-Warning "Neplatna volba '$t'. Zadej 1, 2 nebo 3."
    }
}

function Read-GpuCount {
    param([int]$Detected, [string]$Type)
    $label = Get-GpuTypeLabel $Type
    $counts = Get-GpuTypeCounts $Type
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════"
    Write-Host "  Kolik NVIDIA $label GPU mas k dispozici pro pipeline?"
    Write-Host "  Podporovane hodnoty: $($counts -join ', ')"
    if ($Type -eq "rtxpro6000") {
        Write-Host "  (min. 2 - Qwen 122B v NVFP4 se na jednu 96GB kartu nevejde)"
    }
    if ($Type -eq "a100") {
        Write-Host "  (min. 4 - Qwen 122B v INT4 ~65 GB + KV cache; 4x40 GB = 160 GB)"
    }
    Write-Host "  (jen power-of-2 - TP=3/5/6/7 SGLang/vLLM nepodporuji)"
    if ($Detected -gt 0) {
        Write-Host "  -> Detekovano $Detected GPU na hostu"
    }
    Write-Host "════════════════════════════════════════════════════════"
    while ($true) {
        $n = Read-Host "Zadej pocet $label GPU [$($counts -join '/')]"
        if ($n -match '^\d+$' -and [int]$n -in $counts) {
            $nInt = [int]$n
            if ($Detected -gt 0 -and $nInt -gt $Detected) {
                Write-Warning "Host ma jen $Detected GPU, zadano $nInt. Vyber max $Detected."
                continue
            }
            return $nInt
        }
        Write-Warning "Neplatna hodnota '$n'. Zadej $($counts -join ', ')."
    }
}

# Phase=finalize je CPU only — preflighty + prompt nepotrebujeme.
if ($Phase -ne "finalize") {
    Test-QuantizationPreflight
    Test-HfTokenPreflight

    $detected = Get-DetectedGpuCount
    if ($detected -gt 0) {
        Write-Log "Detekovano $detected GPU na hostu (pres nvidia-smi)"
    } else {
        Write-Warn "nvidia-smi nedostupne - nemuzu detekovat GPU count automaticky"
    }

    # Typ GPU: -GpuType > (s -GpuCount bez promptu) nvidia-smi / b200 > prompt
    if (-not $GpuType) {
        $detectedType = Get-DetectedGpuType
        if ($GpuCount -ne 0) {
            $GpuType = if ($detectedType) { $detectedType } else { "b200" }
        } else {
            $GpuType = Read-GpuType -Detected $detectedType
        }
    }
    $gpuLabel = Get-GpuTypeLabel $GpuType
    $validCounts = Get-GpuTypeCounts $GpuType
    Write-Log "Typ GPU: $gpuLabel"

    if ($detected -gt 0 -and $detected -lt $validCounts[0]) {
        Write-Err "Host ma jen $detected GPU, ale $gpuLabel potrebuje min. $($validCounts[0])"
        Write-Err "(Qwen3.5-122B se nevejde: NVFP4 ~80 GB / INT4 ~65 GB vah + KV cache)."
        throw "GPU count mismatch"
    }

    if ($GpuCount -eq 0) {
        $GpuCount = Read-GpuCount -Detected $detected -Type $GpuType
    } elseif ($GpuCount -notin $validCounts) {
        Write-Err "-GpuCount $GpuCount neni pro $gpuLabel podporovany. Validni: $($validCounts -join ', ')."
        if ($GpuType -eq "rtxpro6000") {
            Write-Err "RTX PRO 6000 ma 96 GB - Qwen3.5-122B-NVFP4 se na jednu kartu nevejde."
        }
        if ($GpuType -eq "a100") {
            Write-Err "A100 ma 40 GB - Qwen3.5-122B v INT4 (~65 GB) potrebuje min. 4 karty."
        }
        exit 1
    } elseif ($detected -gt 0 -and $GpuCount -gt $detected) {
        Write-Err "GpuCount=$GpuCount, ale host ma jen $detected GPU."
        Write-Err "Docker by selhal s 'nvidia-container-cli: device error'."
        throw "GPU count mismatch"
    }
    # Auto-config .env podle GpuType + GpuCount — pro docker compose i native
    Set-EnvForGpuCount -Count $GpuCount -Type $GpuType
    # Propagace do env pro docker compose
    if (-not $env:EXTRACTOR_TP_SIZE) { $env:EXTRACTOR_TP_SIZE = "$GpuCount" }
    # CRITIC_TP_SIZE urcuje HW preset v .env (A100 -> 2, Blackwell -> 1);
    # explicitni $env: override ma prednost.
    if (-not $env:CRITIC_TP_SIZE) {
        $criticTpFile = if ($env:ENV_FILE) { $env:ENV_FILE } else { ".env" }
        $criticTpEnv = Read-EnvVarValue -File $criticTpFile -Key "CRITIC_TP_SIZE"
        if ($criticTpEnv) { $env:CRITIC_TP_SIZE = $criticTpEnv } else { $env:CRITIC_TP_SIZE = "1" }
    }
    if ([int]$env:EXTRACTOR_TP_SIZE -lt $validCounts[0]) {
        Write-Err "EXTRACTOR_TP_SIZE=$($env:EXTRACTOR_TP_SIZE) (z env) - na $gpuLabel musi byt min. $($validCounts[0])."
        throw "EXTRACTOR_TP_SIZE too small"
    }
    Write-Host ""
    Write-Host "Docker GPU config: EXTRACTOR_TP_SIZE=$($env:EXTRACTOR_TP_SIZE) CRITIC_TP_SIZE=$($env:CRITIC_TP_SIZE)"
}

# ---------------------------------------------------------------------------
# Pocet ukazek pro NAVRH TAXONOMIE (TAXONOMY_SAMPLES_PER_DATASET: auto | cislo)
# ---------------------------------------------------------------------------
# auto  = kazda domena si pocet spocita sama z rozmanitosti obsahu (embeddingy
#         bge-m3, faze 0a "sample"); cislo = presne tolik ukazek per domena.
# -TaxonomySamples / $env:TAXONOMY_SAMPLES_PER_DATASET dotaz preskoci; default
# = hodnota z .env, jinak auto. Bez terminalu se pouzije default.
function Test-TaxonomySamplesValue {
    param([string]$Value)
    return ($Value -eq "auto" -or $Value -match '^[1-9]\d*$')
}

function Get-DotEnvValue {
    param([string]$Key)
    $envFile = if ($env:ENV_FILE) { $env:ENV_FILE } else { ".env" }
    if (-not (Test-Path $envFile)) { return "" }
    $val = ""
    foreach ($line in (Get-Content $envFile)) {
        if ($line -match "^[ \t]*$Key=(.*)$") { $val = $Matches[1].Trim().Trim('"').Trim("'") }
    }
    return $val
}

function Resolve-TaxonomySamples {
    param([string]$Given)
    $v = if ($Given) { $Given } elseif ($env:TAXONOMY_SAMPLES_PER_DATASET) { $env:TAXONOMY_SAMPLES_PER_DATASET } else { "" }
    $v = $v.Trim().ToLower()
    if ($v) {
        if (-not (Test-TaxonomySamplesValue $v)) {
            Write-Err "TAXONOMY_SAMPLES_PER_DATASET='$v' musi byt 'auto' nebo kladne cele cislo."
            exit 1
        }
        Write-Log "Ukazky pro taxonomii: $v"
        return $v
    }
    $default = (Get-DotEnvValue "TAXONOMY_SAMPLES_PER_DATASET").ToLower()
    if (-not (Test-TaxonomySamplesValue $default)) { $default = "auto" }
    if ([Console]::IsInputRedirected) {
        Write-Log "Neinteraktivni beh -> ukazky pro taxonomii = $default (default)"
        return $default
    }
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════"
    Write-Host "  Kolik ukazek per domena ma dostat LLM pro NAVRH TAXONOMIE?"
    Write-Host "    auto  = kazda domena si pocet spocita sama podle rozmanitosti"
    Write-Host "            obsahu (embeddingy BAAI/bge-m3)"
    Write-Host "    cislo = presne tolik ukazek per domena (co nejruznejsi texty)"
    Write-Host "════════════════════════════════════════════════════════"
    while ($true) {
        $n = Read-Host "Ukazek pro taxonomii per domena [auto/cislo] (Enter = $default)"
        if (-not $n) { $n = $default }
        $n = $n.Trim().ToLower()
        if (Test-TaxonomySamplesValue $n) {
            Write-Log "OK: ukazky pro taxonomii = $n"
            return $n
        }
        Write-Warning "Neplatna hodnota '$n'. Zadej 'auto' nebo kladne cele cislo."
    }
}

# ---------------------------------------------------------------------------
# Pořadí fází + spuštění od konkrétní fáze (from-<fáze>)
# ---------------------------------------------------------------------------
# .\scripts\run_production.ps1 from-extract  spusti extract -> critic ->
# revise -> finalize. Faze 0 se v takovem behu NEspousti - extract si nacte
# existujici checkpoints\entity_taxonomy.json (viz EXTRACT_EXTRA_ARGS nize).
$PipelineOrder = @("sample", "taxonomy", "extract", "critic", "revise", "finalize")

function Get-PhaseIndex {
    param([string]$Name)
    return [array]::IndexOf($PipelineOrder, $Name)
}

function Get-PhasesFrom {
    param([string]$Name)
    $i = Get-PhaseIndex $Name
    if ($i -lt 0) { return @() }
    return $PipelineOrder[$i..($PipelineOrder.Count - 1)]
}

$FromPhase = ""
if ($Phase -like "from-*") {
    $FromPhase = $Phase.Substring(5)
    if ((Get-PhaseIndex $FromPhase) -lt 0) {
        Write-Err "from-${FromPhase}: neznama faze. Validni: $($PipelineOrder -join ', ')."
        exit 1
    }
}

# Prvni faze behu - podle ni se rozhoduje, co se pta a co se preskoci.
$StartPhase = $Phase
if ($Phase -eq "all") { $StartPhase = "sample" }
if ($FromPhase) { $StartPhase = $FromPhase }

function Test-RunHasPhase {
    param([string]$Name)
    if ($Phase -eq $Name) { return $true }
    if ($Phase -eq "all" -or $FromPhase) {
        return ((Get-PhaseIndex $Name) -ge (Get-PhaseIndex $StartPhase))
    }
    return $false
}

# Volitelny dotaz na soubeznost a davky. Enter = hodnoty z .env (resp. HW
# preset). Vypnout: $env:THROUGHPUT_PROMPT="0". Neinteraktivni beh se nepta.
#   *_CONCURRENCY = kolik requestu posila labeler soucasne
#   *_BATCH_SIZE  = kolik jich server pocita soucasne (max-running-requests)
# Batch pamet nealokuje (KV pool urcuje *_MEM_FRACTION); optimum zmeri
# ./scripts/auto_tune_concurrency.sh.
function Read-PositiveInt {
    param([string]$Label, [string]$Key, [string]$Current)
    $envFile = if ($env:ENV_FILE) { $env:ENV_FILE } else { ".env" }
    while ($true) {
        $n = Read-Host "  $Label [$Current]"
        if (-not $n) { $n = $Current }
        if ($n -match '^[1-9][0-9]*$') {
            Update-EnvVar -File $envFile -Key $Key -Value $n
            Set-Item -Path "env:$Key" -Value $n
            return
        }
        Write-Warning "Zadej kladne cele cislo."
    }
}

function Read-Throughput {
    if ($env:THROUGHPUT_PROMPT -eq "0") { return }
    if ([Console]::IsInputRedirected) { return }
    $envFile = if ($env:ENV_FILE) { $env:ENV_FILE } else { ".env" }
    $ec = if ($env:EXTRACTOR_CONCURRENCY) { $env:EXTRACTOR_CONCURRENCY } else { Read-EnvVarValue -File $envFile -Key "EXTRACTOR_CONCURRENCY" }
    $eb = if ($env:EXTRACTOR_BATCH_SIZE)  { $env:EXTRACTOR_BATCH_SIZE }  else { Read-EnvVarValue -File $envFile -Key "EXTRACTOR_BATCH_SIZE" }
    $cc = if ($env:CRITIC_CONCURRENCY)    { $env:CRITIC_CONCURRENCY }    else { Read-EnvVarValue -File $envFile -Key "CRITIC_CONCURRENCY" }
    $cb = if ($env:CRITIC_BATCH_SIZE)     { $env:CRITIC_BATCH_SIZE }     else { Read-EnvVarValue -File $envFile -Key "CRITIC_BATCH_SIZE" }
    if (-not $ec) { $ec = "32" }
    if (-not $eb) { $eb = "32" }
    if (-not $cc) { $cc = "64" }
    if (-not $cb) { $cb = "64" }
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════"
    Write-Host "  Soubeznost a davky (aktualne z .env):"
    Write-Host "    extractor: concurrency $ec / server batch $eb"
    Write-Host "    critic:    concurrency $cc / server batch $cb"
    Write-Host "  concurrency = kolik requestu posila labeler soucasne"
    Write-Host "  batch       = kolik jich server pocita soucasne"
    Write-Host "  (optimum zmeri .\scripts\auto_tune_concurrency.sh)"
    Write-Host "════════════════════════════════════════════════════════"
    $yn = Read-Host "Upravit tyhle hodnoty? [a/N] (Enter = nechat)"
    if ($yn -notmatch '^(a|A|y|Y|ano|Ano|yes|Yes)$') {
        Write-Log "Soubeznost zustava: extractor $ec/$eb, critic $cc/$cb"
        return
    }
    Read-PositiveInt -Label "EXTRACTOR_CONCURRENCY" -Key "EXTRACTOR_CONCURRENCY" -Current $ec
    Read-PositiveInt -Label "EXTRACTOR_BATCH_SIZE " -Key "EXTRACTOR_BATCH_SIZE"  -Current $eb
    Read-PositiveInt -Label "CRITIC_CONCURRENCY   " -Key "CRITIC_CONCURRENCY"    -Current $cc
    Read-PositiveInt -Label "CRITIC_BATCH_SIZE    " -Key "CRITIC_BATCH_SIZE"     -Current $cb
    Write-Log "OK: extractor $($env:EXTRACTOR_CONCURRENCY)/$($env:EXTRACTOR_BATCH_SIZE), critic $($env:CRITIC_CONCURRENCY)/$($env:CRITIC_BATCH_SIZE)"
}

# Pocet ukazek pro taxonomii se tyka fazi sample (0a) a taxonomy (0)
if ((Test-RunHasPhase "sample") -or (Test-RunHasPhase "taxonomy")) {
    $env:TAXONOMY_SAMPLES_PER_DATASET = Resolve-TaxonomySamples -Given $TaxonomySamples
}
# Soubeznost / davky - jen kdyz bezi nektera LLM faze
if ((Test-RunHasPhase "extract") -or (Test-RunHasPhase "critic") -or (Test-RunHasPhase "revise")) {
    Read-Throughput
}

function Test-CudaPreflight {
    # PS 5.1: redirect stderr nativniho prikazu (2>$null / 2>&1) bali radky
    # do ErrorRecordu; s globalnim $ErrorActionPreference=Stop by PRVNI progress
    # hlaska dockeru (jde na stderr) shodila skript. Lokalni Continue plati
    # jen pro tuto funkci.
    $ErrorActionPreference = "Continue"
    if ($env:SKIP_CUDA_CHECK -eq "1") {
        Write-Warn "SKIP_CUDA_CHECK=1 — preskakuji CUDA verifikaci."
        return
    }
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        Write-Warn "nvidia-smi nenalezen — nemuzu verifikovat CUDA verzi."
        Write-Warn "Pokud kontejnery selhou s 'no kernel image' chybou, ujisti se,"
        Write-Warn "ze mas NVIDIA driver >= 580 (CUDA 13.0 capable)."
        return
    }
    $smiOutput = & nvidia-smi 2>$null | Out-String
    $match = [regex]::Match($smiOutput, "CUDA Version:\s*(\d+)\.(\d+)")
    if (-not $match.Success) {
        Write-Warn "Nelze ziskat CUDA verzi z nvidia-smi vystupu — pokracuju."
        return
    }
    $major = [int]$match.Groups[1].Value
    $minor = [int]$match.Groups[2].Value
    # CUDA 13+ je potreba jen pro NVFP4 na Blackwellu; A100 (Ampere) jede
    # INT4/FP8 pres Marlin, kde staci CUDA 12.1+.
    $minMajor = 13
    if ($GpuType -eq "a100") { $minMajor = 12 }
    if ($major -lt $minMajor) {
        if ($minMajor -eq 12) {
            Write-Err "Detekovana CUDA $major.$minor - na A100 je potreba CUDA 12.1+ (Marlin INT4/FP8)."
        } else {
            Write-Err "Detekovana CUDA $major.$minor - pipeline VYZADUJE CUDA 13+ pro NVFP4"
            Write-Err "na Blackwell SM120. Upgrade NVIDIA driveru na >=580 a CUDA toolkit >=13.0."
        }
        Write-Err "Pokud chces pokracovat na vlastni riziko (napr. testujes FP8/BF16 fallback),"
        Write-Err "spust: `$env:SKIP_CUDA_CHECK='1'; .\scripts\run_production.ps1 $Phase"
        throw "CUDA ${minMajor}+ required (detected $major.$minor)"
    }
    Write-Log "Preflight OK — CUDA $major.$minor detekovana (>=13.0)."
}

if ($Phase -ne "finalize") {
    Test-CudaPreflight
}

function Try-Profile {
    param([string]$Profile, [string]$LabelerService)
    # PS 5.1: redirect stderr nativniho prikazu (2>$null / 2>&1) bali radky
    # do ErrorRecordu; s globalnim $ErrorActionPreference=Stop by PRVNI progress
    # hlaska dockeru (jde na stderr) shodila skript. Lokalni Continue plati
    # jen pro tuto funkci.
    $ErrorActionPreference = "Continue"
    Write-Log "Zkousim profil: $Profile (detached -d, na pozadi)"
    docker compose --profile $Profile up -d --build
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Profile ${Profile}: docker compose up -d selhal"
        docker compose --profile $Profile down 2>&1 | Out-Null
        return $false
    }
    Write-Log "Containery bezi na pozadi. Sleduj progress v separatnim terminalu:"
    Write-Log "  docker compose --profile $Profile logs -f --no-log-prefix $LabelerService"
    # Cekej na labeler exit code
    docker compose --profile $Profile wait $LabelerService 2>&1 | Out-Null
    $rc = $LASTEXITCODE
    # `docker compose wait` se mezi verzemi lisi (vraci exit code kontejneru
    # jako svuj status a na stdout pise vetu) -> cislo bereme z inspectu.
    $cidWait = (docker compose --profile $Profile ps -aq $LabelerService 2>$null | Select-Object -First 1)
    if ($cidWait) {
        $codeWait = (docker inspect --format='{{.State.ExitCode}}' $cidWait 2>$null | Select-Object -First 1)
        if ($codeWait -match '^[0-9]+$') { $rc = [int]$codeWait }
    }
    if ($rc -ne 0) {
        # Fallback: ps -q + docker inspect
        $cid = (docker compose --profile $Profile ps -q $LabelerService 2>$null | Select-Object -First 1)
        if ($cid) {
            docker wait $cid 2>&1 | Out-Null
            $rc = [int](docker inspect --format='{{.State.ExitCode}}' $cid 2>$null)
        }
    }
    docker compose --profile $Profile down 2>&1 | Out-Null
    if ($rc -eq 0) {
        Write-Log "Profile $Profile hotov +"
        return $true
    }
    Write-Err "Profile $Profile selhal (rc=$rc)"
    return $false
}

# ---------------------------------------------------------------------------
# NATIVE runtime helpers — bez dockeru, Python proti bezicimu LLM serveru
# ---------------------------------------------------------------------------
$NativeAutoLlm = if ($env:NATIVE_AUTO_LLM) { $env:NATIVE_AUTO_LLM } else { "1" }

function Set-NativeEnvParity {
    if ($Runtime -ne "native") { return }
    if (-not $env:HF_XET_HIGH_PERFORMANCE)           { $env:HF_XET_HIGH_PERFORMANCE = "1" }
    if (-not $env:PYTHONWARNINGS)                    { $env:PYTHONWARNINGS = "ignore::DeprecationWarning,ignore::FutureWarning" }
    if (-not $env:TRANSFORMERS_VERBOSITY)            { $env:TRANSFORMERS_VERBOSITY = "error" }
    if (-not $env:TRANSFORMERS_NO_ADVISORY_WARNINGS) { $env:TRANSFORMERS_NO_ADVISORY_WARNINGS = "1" }
    if (-not $env:EXTRACTOR_BASE_URL)                { $env:EXTRACTOR_BASE_URL = "http://localhost:8000/v1" }
    if (-not $env:CRITIC_BASE_URL)                   { $env:CRITIC_BASE_URL = "http://localhost:8001/v1" }
    if (-not $env:REVISOR_BASE_URL)                  { $env:REVISOR_BASE_URL = $env:EXTRACTOR_BASE_URL }
    # Embedding server faze 0a (vLLM --runner pooling)
    if (-not $env:EMBEDDING_BASE_URL)                { $env:EMBEDDING_BASE_URL = "http://localhost:8002/v1" }
    Write-Log "Native env parita: HF_XET=1, PYTHONWARNINGS=suppress"
    Write-Host "  EXTRACTOR_BASE_URL=$($env:EXTRACTOR_BASE_URL)"
    Write-Host "  CRITIC_BASE_URL=$($env:CRITIC_BASE_URL)"
}

function Get-SglangArgsForPhase {
    param([string]$Phase)
    $args = @()
    switch ($Phase) {
        { $_ -in "extract", "revise" } {
            $batch = if ($env:EXTRACTOR_BATCH_SIZE) { $env:EXTRACTOR_BATCH_SIZE } else { "32" }
            $ctx   = if ($env:EXTRACTOR_CONTEXT_LEN) { $env:EXTRACTOR_CONTEXT_LEN } else { "65536" }
            $args  = @("--max-running-requests=$batch", "--context-length=$ctx", "--reasoning-parser=qwen3")
            if ($env:EXTRACTOR_QUANTIZATION) { $args += "--quantization=$($env:EXTRACTOR_QUANTIZATION)" }
            # Mamba state pool - u hybridniho Qwen3.5 dela strop soubeznosti on,
            # ne KV cache (viz komentar v docker-compose.yml u extractor-sglang).
            if ($env:EXTRACTOR_MAX_MAMBA_CACHE) { $args += "--max-mamba-cache-size=$($env:EXTRACTOR_MAX_MAMBA_CACHE)" }
            if ($env:EXTRACTOR_MAMBA_SSM_DTYPE) { $args += "--mamba-ssm-dtype=$($env:EXTRACTOR_MAMBA_SSM_DTYPE)" }
            if ($env:EXTRACTOR_MAMBA_RADIX_STRATEGY) { $args += "--mamba-radix-cache-strategy=$($env:EXTRACTOR_MAMBA_RADIX_STRATEGY)" }
            $eff = if ($env:EXTRACTOR_DISABLE_PIECEWISE_CUDA_GRAPH) { $env:EXTRACTOR_DISABLE_PIECEWISE_CUDA_GRAPH } else { $env:DISABLE_PIECEWISE_CUDA_GRAPH }
            if ($eff -eq "1" -or $eff -eq "true") { $args += "--disable-piecewise-cuda-graph" }
        }
        "critic" {
            $batch = if ($env:CRITIC_BATCH_SIZE) { $env:CRITIC_BATCH_SIZE } else { "64" }
            $ctx   = if ($env:CRITIC_CONTEXT_LEN) { $env:CRITIC_CONTEXT_LEN } else { "65536" }
            $args  = @("--max-running-requests=$batch", "--context-length=$ctx")
            if ($env:CRITIC_QUANTIZATION) { $args += "--quantization=$($env:CRITIC_QUANTIZATION)" }
            $eff = if ($env:CRITIC_DISABLE_PIECEWISE_CUDA_GRAPH) { $env:CRITIC_DISABLE_PIECEWISE_CUDA_GRAPH } else { $env:DISABLE_PIECEWISE_CUDA_GRAPH }
            if ($eff -eq "1" -or $eff -eq "true") { $args += "--disable-piecewise-cuda-graph" }
        }
    }
    return ($args -join ' ')
}

Set-NativeEnvParity

function Test-NativeLlmEndpoint {
    param([string]$Url)
    try {
        $resp = Invoke-WebRequest -Uri "$($Url.TrimEnd('/'))/models" `
            -Method Get -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        return ($resp.StatusCode -lt 400)
    } catch { return $false }
}

function Install-SgLangIfMissing {
    # PS 5.1: redirect stderr nativniho prikazu (2>$null / 2>&1) bali radky
    # do ErrorRecordu; s globalnim $ErrorActionPreference=Stop by PRVNI progress
    # hlaska dockeru (jde na stderr) shodila skript. Lokalni Continue plati
    # jen pro tuto funkci.
    $ErrorActionPreference = "Continue"
    try {
        python -c "import sglang" 2>$null
        if ($LASTEXITCODE -eq 0) { return $true }
    } catch {}
    Write-Warn "SGLang nenalezen."
    if ($NativeAutoLlm -ne "1") {
        Write-Err "NATIVE_AUTO_LLM=0 - rucne: .\scripts\install_sglang.ps1"
        return $false
    }
    Write-Log "Auto-install SGLang..."
    & "$PSScriptRoot\install_sglang.ps1"
    return ($LASTEXITCODE -eq 0)
}

function Start-NativeLlmForPhase {
    param([string]$Phase, [int]$Port)
    if (Test-NativeLlmEndpoint -Url "http://localhost:$Port/v1") {
        Write-Log "LLM server na portu $Port uz bezi"
        return $true
    }
    if ($NativeAutoLlm -ne "1") {
        Write-Err "Endpoint http://localhost:$Port/v1 NEODPOVIDA. Manual: spust LLM:"
        Write-Err "  .\scripts\start_native_llm.ps1 <model> $Port $($env:EXTRACTOR_TP_SIZE)"
        return $false
    }
    $model = ""
    $tp = 1
    switch ($Phase) {
        { $_ -in "extract", "revise" } {
            $model = if ($env:EXTRACTOR_MODEL) { $env:EXTRACTOR_MODEL } else { "RedHatAI/Qwen3.5-122B-A10B-NVFP4" }
            $tp = if ($env:EXTRACTOR_TP_SIZE) { [int]$env:EXTRACTOR_TP_SIZE } else { 1 }
        }
        "critic" {
            $model = if ($env:CRITIC_MODEL) { $env:CRITIC_MODEL } else { "nvidia/Gemma-4-31B-IT-NVFP4" }
            $tp = if ($env:CRITIC_TP_SIZE) { [int]$env:CRITIC_TP_SIZE } else { 1 }
        }
        default { return $true }
    }
    # Args 1:1 jako Docker
    $extra = Get-SglangArgsForPhase -Phase $Phase
    Write-Log "Auto-start SGLang pro '$Phase': $model na portu $Port (tp=$tp)"
    Write-Log "  Args: $extra"
    & "$PSScriptRoot\start_native_llm.ps1" -ModelPath $model -Port $Port -TpSize $tp -ExtraArgs $extra
    return ($LASTEXITCODE -eq 0)
}

function Stop-NativeLlmForPhase {
    param([int]$Port)
    if ($NativeAutoLlm -ne "1") { return }
    $LogDir = if ($env:SGLANG_LOG_DIR) { $env:SGLANG_LOG_DIR } else { $env:TEMP }
    if (Test-Path (Join-Path $LogDir "sglang_$Port.pid")) {
        Write-Log "Auto-stop SGLang na portu $Port"
        & "$PSScriptRoot\stop_native_llm.ps1" -Port $Port
    }
}

# Cleanup hook — Ctrl+C / fail / normal exit → zastav vsechny SGLang servery.
# Iteruje pres PID files (sglang_*.pid), ne hardcoded ports.
function Invoke-NativeCleanup {
    if ($Runtime -ne "native" -or $NativeAutoLlm -ne "1") { return }
    $LogDir = if ($env:SGLANG_LOG_DIR) { $env:SGLANG_LOG_DIR } else { $env:TEMP }
    $pidFiles = Get-ChildItem -Path $LogDir -Filter "sglang_*.pid" -ErrorAction SilentlyContinue
    foreach ($pf in $pidFiles) {
        $port = $pf.BaseName -replace '^sglang_', ''
        Write-Warn "Cleanup: stop SGLang na portu $port (exit)"
        try {
            & "$PSScriptRoot\stop_native_llm.ps1" -Port ([int]$port) 2>&1 | Out-Null
        } catch {}
    }
}
Register-EngineEvent PowerShell.Exiting -Action { Invoke-NativeCleanup } | Out-Null

function Run-NativePhase {
    param([string]$Ph)
    Write-Log "FAZE [$Ph] — NATIVE (Python primo, bez dockeru)"

    if ($Ph -ne "finalize") {
        if (-not (Install-SgLangIfMissing)) { throw "$($Ph): SGLang missing/install failed" }
        $port = switch ($Ph) { "taxonomy" { 8000 } "extract" { 8000 } "critic" { 8001 } "revise" { 8000 } }
        if (-not (Start-NativeLlmForPhase -Phase $Ph -Port $port)) {
            throw "$($Ph): LLM endpoint nedostupny / auto-start selhal"
        }
    }

    if ($Ph -eq "taxonomy") {
        python -m labeler.main taxonomy --mode auto
    } else {
        python -m labeler.main $Ph
    }
    $rc = $LASTEXITCODE

    if ($Ph -ne "finalize") {
        $port = switch ($Ph) { "taxonomy" { 8000 } "extract" { 8000 } "critic" { 8001 } "revise" { 8000 } }
        Stop-NativeLlmForPhase -Port $port
    }

    if ($rc -ne 0) { throw "$Ph selhal (native, rc=$rc)" }
    Write-Log "OK $Ph hotov (native)"
}

function Run-Taxonomy {
    # PS 5.1: stderr nativnich prikazu (docker progress) nesmi shodit skript.
    $ErrorActionPreference = "Continue"
    Write-Log "FAZE 0 - TAXONOMY (auto-discovery, jednotna taxonomie napric datasety)"
    if ($Runtime -eq "native") { Run-NativePhase "taxonomy"; return }
    # Taxonomy bezi na EXTRACTOR LLM (Qwen). `run` s override commandu na
    # `taxonomy --mode auto`, foreground -> primy exit code.
    foreach ($backend in $ExtractorBackends) {
        Write-Log "Zkousim profil: extract-$backend (taxonomy)"
        docker compose --profile "extract-$backend" run --rm "labeler-extract-$backend" taxonomy --mode auto
        $rc = $LASTEXITCODE
        docker compose --profile "extract-$backend" down 2>$null | Out-Null
        if ($rc -eq 0) {
            Write-Log "OK Taxonomy hotova pres backend: $backend"
            return
        }
        Write-Warn "Backend $backend selhal pro taxonomy (rc=$rc), zkousim dalsi..."
    }
    # Faze 0 je best-effort - bez taxonomie pipeline pojede s defaultnimi
    # 16 kategoriemi, takze beh nezastavujeme.
    Write-Warn "Taxonomy neprosla na zadnem backendu - pipeline pojede s defaultnimi 16 kategoriemi."
}

function Run-Sample {
    # PS 5.1: stderr nativnich prikazu (docker progress) nesmi shodit skript.
    $ErrorActionPreference = "Continue"
    Write-Log "FAZE 0a - SAMPLE (vyber ukazek pro taxonomii, embeddingy bge-m3 na vLLM)"
    if ($Runtime -eq "native") {
    # Embedding server faze 0a bezi na vLLM, ktery na Windows nativne nebezi
    # (jen Linux) -> v native rezimu se faze preskoci; taxonomie pouzije
    # nahodny vyber. Pro embeddingy pust Docker runtime (nebo WSL + run_*.sh).
        Write-Warn "Native runtime: vLLM na Windows nebezi - faze sample preskocena, taxonomie pouzije nahodny vyber."
        return
    }
    # Stejne jako ostatni faze: detached (-d), progress se cte z logu.
    $gpus = if ($env:EMBEDDING_GPU_COUNT) { $env:EMBEDDING_GPU_COUNT } else { "1" }
    Write-Log "Embedder na ${gpus}x GPU; progress:"
    Write-Log "  docker compose --profile embed-vllm logs -f --no-log-prefix labeler-embed-vllm"
    docker compose --profile embed-vllm up -d
    $ok = $false
    if ($LASTEXITCODE -eq 0) {
        docker compose --profile embed-vllm wait labeler-embed-vllm 2>&1 | Out-Null
        $cid = (docker compose --profile embed-vllm ps -aq labeler-embed-vllm 2>$null | Select-Object -First 1)
        $code = if ($cid) { (docker inspect --format='{{.State.ExitCode}}' $cid 2>$null | Select-Object -First 1) } else { "" }
        $ok = ($code -match '^[0-9]+$' -and [int]$code -eq 0)
    }
    docker compose --profile embed-vllm down 2>$null | Out-Null
    if ($ok) {
        Write-Log "OK Ukazky pro taxonomii vybrany (checkpoints\taxonomy_sampling.json)"
    } else {
        Write-Warn "Vyber ukazek pres embeddingy selhal - taxonomie pouzije nahodny vyber."
    }
}

function Run-Extract {
    Write-Log "FÁZE 1 — EXTRACT (Qwen3.5-122B-A10B)"
    if ($Runtime -eq "native") { Run-NativePhase "extract"; return }
    foreach ($backend in $ExtractorBackends) {
        if (Try-Profile "extract-$backend" "labeler-extract-$backend") {
            Write-Log "OK Extract hotov přes backend: $backend"
            return
        }
        Write-Err "Backend $backend selhal, zkouším další..."
    }
    throw "VŠECHNY extract backendy selhaly."
}

function Run-Critic {
    Write-Log "FÁZE 2 — CRITIC (Gemma 4 31B, sám na GPU)"
    $ckpt = if ($env:CHECKPOINT_DIR) { $env:CHECKPOINT_DIR } else { ".\checkpoints" }
    if (-not (Test-Path "$ckpt\extractions.jsonl")) {
        throw "Chybí $ckpt\extractions.jsonl — nejdřív pusť extract fázi"
    }
    if ($Runtime -eq "native") { Run-NativePhase "critic"; return }
    foreach ($backend in $CriticBackends) {
        if (Try-Profile "critic-$backend" "labeler-critic-$backend") {
            Write-Log "OK Critic hotov přes backend: $backend"
            return
        }
        Write-Err "Backend $backend selhal, zkouším další..."
    }
    throw "VŠECHNY critic backendy selhaly."
}

function Run-Revise {
    Write-Log "FÁZE 2b — REVISE (Qwen revisor projde non-ok kritiky, sám na GPU)"
    $ckpt = if ($env:CHECKPOINT_DIR) { $env:CHECKPOINT_DIR } else { ".\checkpoints" }
    if (-not (Test-Path "$ckpt\critic_results.jsonl")) {
        throw "Chybí $ckpt\critic_results.jsonl — nejdřív pusť critic fázi"
    }
    if ($Runtime -eq "native") { Run-NativePhase "revise"; return }
    foreach ($backend in $ReviseBackends) {
        if (Try-Profile "revise-$backend" "labeler-revise-$backend") {
            Write-Log "OK Revise hotov přes backend: $backend"
            return
        }
        Write-Err "Backend $backend selhal, zkouším další..."
    }
    throw "VŠECHNY revise backendy selhaly."
}

function Run-Finalize {
    Write-Log "FAZE 3 — FINALIZE (merge + stats, bez LLM, detached -d)"
    if ($Runtime -eq "native") { Run-NativePhase "finalize"; return }
    docker compose --profile finalize up -d --build
    if ($LASTEXITCODE -ne 0) {
        docker compose --profile finalize down 2>&1 | Out-Null
        throw "Finalize: docker compose up -d selhal"
    }
    Write-Log "Container bezi na pozadi. Sleduj: docker compose --profile finalize logs -f labeler-finalize"
    docker compose --profile finalize wait labeler-finalize 2>&1 | Out-Null
    $rc = $LASTEXITCODE
    docker compose --profile finalize down 2>&1 | Out-Null
    if ($rc -ne 0) { throw "Finalize selhal (rc=$rc)" }
    Write-Log "OK Finalize hotov"
}

# ---------------------------------------------------------------------------
# Spusteni od extract dal -> faze 0 se NESPOUSTI
# ---------------------------------------------------------------------------
# Default --taxonomy-mode v labeleru je `auto`, tj. extract by si taxonomii
# navrhl znovu. Kdyz beh zacina na/za extract, uzivatel fazi 0 vedome
# preskakuje -> posleme `--taxonomy-mode off` a extract si nacte existujici
# checkpoints\entity_taxonomy.json (load_taxonomy_if_exists).
if ((Get-PhaseIndex $StartPhase) -ge 2 -and $Phase -ne "finalize") {
    if ($env:EXTRACT_EXTRA_ARGS -notlike "*--taxonomy-mode*") {
        $env:EXTRACT_EXTRA_ARGS = ("--taxonomy-mode off " + $env:EXTRACT_EXTRA_ARGS).Trim()
        $ckptDir = if ($env:CHECKPOINT_DIR) { $env:CHECKPOINT_DIR } else { ".\checkpoints" }
        $taxFile = Join-Path $ckptDir "entity_taxonomy.json"
        if (Test-RunHasPhase "extract") {
            if (Test-Path $taxFile) {
                Write-Log "Faze 0 se preskakuje - extract pouzije existujici $taxFile"
            } else {
                Write-Warn "Faze 0 se preskakuje, ale $taxFile neexistuje - extract pojede s defaultnimi 16 kategoriemi."
                Write-Warn "Taxonomii navrhnes: .\scripts\run_production.ps1 taxonomy"
            }
        }
    }
}


# Spusti zadane faze v poradi; pred extract doplni --taxonomy-mode off, aby
# se discovery nedelala dvakrat (faze 0 uz probehla samostatne).
function Invoke-Phases {
    param([string[]]$Phases)
    foreach ($p in $Phases) {
        if ($p -eq "extract" -and ($env:EXTRACT_EXTRA_ARGS -notlike "*--taxonomy-mode*")) {
            $env:EXTRACT_EXTRA_ARGS = ("--taxonomy-mode off " + $env:EXTRACT_EXTRA_ARGS).Trim()
        }
        switch ($p) {
            "sample"   { Run-Sample }
            "taxonomy" { Run-Taxonomy }
            "extract"  { Run-Extract }
            "critic"   { Run-Critic }
            "revise"   { Run-Revise }
            "finalize" { Run-Finalize }
        }
    }
}

$ckpt = if ($env:CHECKPOINT_DIR) { $env:CHECKPOINT_DIR } else { ".\checkpoints" }
if ($FromPhase) {
    $seq = Get-PhasesFrom $FromPhase
    Write-Log "Bezim od faze '$FromPhase' dal: $($seq -join ' ')"
    Invoke-Phases -Phases $seq
    Write-Log "*** HOTOVO (od $FromPhase). Vysledky v $ckpt\ ***"
} else {
    switch ($Phase) {
        "sample"   { Run-Sample }
        "taxonomy" { Run-Taxonomy }
        "extract"  { Run-Extract }
        "critic"   { Run-Critic }
        "revise"   { Run-Revise }
        "finalize" { Run-Finalize }
        "all"      {
            Invoke-Phases -Phases $PipelineOrder
            Write-Log "*** VŠE HOTOVO. Výsledky v $ckpt\ ***"
            Write-Log "   - final_labels.json + .jsonl            (reálné popisky)"
            Write-Log "   - final_labels_synthetic.json + .jsonl  (syntetické texty)"
            Write-Log "   - stats.json                            (souhrn)"
        }
        default {
            Write-Err "Neznámá fáze: $Phase."
            Write-Err "Jedna faze:  $($PipelineOrder -join ' | ') | all"
            Write-Err "Od faze dal: from-sample | from-taxonomy | from-extract | from-critic | from-revise"
            exit 1
        }
    }
}

