# Production smoke test - empiricky odhad doby behu pipeline na cilovem HW.
#
# Co dela:
#   1. Z kazdeho datasetu v data/ vezme N items (zepta se; default 100)
#   2. Postupne spusti vsechny 4 faze (extract/critic/revise/finalize) na
#      smoke vzorku - vzdy 1 model na GPU, jako v produkci
#   3. Meri wall-clock per faze
#   4. Linearne extrapoluje na full produkcni rozsah (pure empirical -
#      zadne teoreticke scaling factory, zadne GPU-class projekce)
#   5. Vygeneruje report (Markdown + JSON) v checkpoints/smoke/
#
# Pouziti (DOCKER runtime - default):
#   .\scripts\run_smoke_test.ps1                                       # interaktivni prompt
#   .\scripts\run_smoke_test.ps1 -GpuCount 2                           # preskoci prompt (typ z nvidia-smi / B200)
#   .\scripts\run_smoke_test.ps1 -GpuType rtxpro6000 -GpuCount 2       # 2x RTX PRO 6000
#   .\scripts\run_smoke_test.ps1 -SamplesPerDataset 100          # preskoci dotaz na pocet polozek
#   .\scripts\run_smoke_test.ps1 -SmokePhases "extract","critic","finalize"
#
# Pouziti (NATIVE runtime - bez dockeru):
#   # 1) Sam si spust LLM server, napr. v jinem terminalu:
#   #    python -m sglang.launch_server --model-path RedHatAI/Qwen3.5-122B-A10B-NVFP4 ...
#   # 2) Spust skript v native modu:
#   .\scripts\run_smoke_test.ps1 -Runtime native
#   # 3) Pokud mas LLM server jinde nez localhost:8000:
#   $env:EXTRACTOR_BASE_URL = "http://10.0.0.5:8000/v1"
#   $env:CRITIC_BASE_URL = "http://10.0.0.5:8001/v1"
#   .\scripts\run_smoke_test.ps1 -Runtime native
#
# Skript se na zacatku zepta na typ GPU (B200 / RTX PRO 6000) a kolik jich
# mas (power-of-2 pro tensor parallelism MoE; RTX PRO 6000 min. 2 - Qwen 122B
# se na jednu 96GB kartu nevejde). Podle toho nastavi EXTRACTOR_TP_SIZE
# a HW_LABEL pro report. -GpuType / -GpuCount prompt preskoci.

param(
    [int]$GpuCount = 0,                       # 0 = interactive prompt
    [ValidateSet("b200", "rtxpro6000", "a100")]
    [string]$GpuType = "",                    # "" = prompt (s -GpuCount: nvidia-smi, jinak b200)
    [string]$HwLabel = "",                    # auto-derive from GpuCount if empty
    [int]$SamplesPerDataset = 0,              # 0 = interactive prompt (Enter / bez terminalu = 100)
    [string[]]$SmokePhases = @("sample", "taxonomy", "extract", "critic", "revise", "finalize"),
    [string]$TaxonomySamples = "",            # auto | cislo; "" = prompt (Enter = .env / auto)
    [double]$DockerStartupS = 0.0,
    [ValidateSet("docker", "native")]
    [string]$Runtime = "docker"               # docker (default) / native (host Python proti bezicimu LLM serveru)
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "..")

function Write-Log  { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Err  { param($m) Write-Host "`n!! $m" -ForegroundColor Red }
function Write-Warn { param($m) Write-Host "`n!! $m" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# Preflight checks + Interactive GPU prompt
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
        Write-Err "NVFP4 modely maji quantization v config.json - auto-detect."
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
    }
}

# ---------------------------------------------------------------------------
# .env auto-config - po prompt podle GpuCount
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
            Write-Warn "Neexistuje $envFile ani .env.example - auto-config preskocen."
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
#   b200        NVIDIA B200, 180 GB HBM3e - Qwen3.5-122B-NVFP4 (~80 GB vah)
#               se vejde na 1 GPU -> 1, 2, 4 nebo 8 GPU
#   rtxpro6000  NVIDIA RTX PRO 6000 Blackwell, 96 GB GDDR7 - Qwen se na jednu
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

Test-QuantizationPreflight
Test-HfTokenPreflight

$detected = Get-DetectedGpuCount
if ($detected -gt 0) {
    Write-Log "Detekovano $detected GPU na hostu"
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

# Auto-config .env podle GpuType + GpuCount - pro docker compose i native
Set-EnvForGpuCount -Count $GpuCount -Type $GpuType

# Propagace do env pro docker compose
$env:EXTRACTOR_TP_SIZE = "$GpuCount"
# CRITIC_TP_SIZE urcuje HW preset v .env (A100 -> 2, Blackwell -> 1);
# explicitni $env: override ma prednost.
if (-not $env:CRITIC_TP_SIZE) {
    $criticTpFile = if ($env:ENV_FILE) { $env:ENV_FILE } else { ".env" }
    $criticTpEnv = Read-EnvVarValue -File $criticTpFile -Key "CRITIC_TP_SIZE"
    if ($criticTpEnv) { $env:CRITIC_TP_SIZE = $criticTpEnv } else { $env:CRITIC_TP_SIZE = "1" }
}

# ---------------------------------------------------------------------------
# Interactive prompt - pocet polozek per dataset
# ---------------------------------------------------------------------------
# Kolik itemu z kazdeho datasetu projde smoke pipeline. Vic = presnejsi
# extrapolace, ale delsi smoke test. -SamplesPerDataset dotaz preskoci;
# bez terminalu (presmerovany vstup) se pouzije default.
$DefaultSamplesPerDataset = 100

function Read-SamplesPerDataset {
    param([int]$Default)
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════"
    Write-Host "  Kolik polozek z kazdeho datasetu ma smoke test zpracovat?"
    Write-Host "  Vic = presnejsi odhad doby behu produkce, ale delsi smoke test."
    Write-Host "════════════════════════════════════════════════════════"
    while ($true) {
        $n = Read-Host "Pocet polozek per dataset (Enter = $Default)"
        if (-not $n) { return $Default }
        if ($n -match '^[1-9]\d*$') { return [int]$n }
        Write-Warning "Neplatna hodnota '$n'. Zadej kladne cele cislo."
    }
}

if ($SamplesPerDataset -lt 0) {
    Write-Err "-SamplesPerDataset musi byt kladne cele cislo (dostal $SamplesPerDataset)."
    exit 1
} elseif ($SamplesPerDataset -eq 0) {
    if ([Console]::IsInputRedirected) {
        $SamplesPerDataset = $DefaultSamplesPerDataset
        Write-Log "Neinteraktivni beh -> SamplesPerDataset=$SamplesPerDataset (default)"
    } else {
        $SamplesPerDataset = Read-SamplesPerDataset -Default $DefaultSamplesPerDataset
        Write-Log "OK: $SamplesPerDataset polozek per dataset"
    }
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

$env:TAXONOMY_SAMPLES_PER_DATASET = Resolve-TaxonomySamples -Given $TaxonomySamples

if ([string]::IsNullOrWhiteSpace($HwLabel)) {
    $HwLabel = "${GpuCount}x $gpuLabel"
}

$ExtractorBackends = if ($env:EXTRACTOR_BACKENDS) { $env:EXTRACTOR_BACKENDS -split ' ' } else { @("sglang", "vllm") }
$CriticBackends    = if ($env:CRITIC_BACKENDS)    { $env:CRITIC_BACKENDS    -split ' ' } else { @("sglang", "vllm") }
$ReviseBackends    = if ($env:REVISE_BACKENDS)    { $env:REVISE_BACKENDS    -split ' ' } else { @("sglang", "vllm") }


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
function Test-CudaPreflight {
    # PS 5.1: redirect stderr nativniho prikazu (2>$null / 2>&1) bali radky
    # do ErrorRecordu; s globalnim $ErrorActionPreference=Stop by PRVNI progress
    # hlaska dockeru (jde na stderr) shodila skript. Lokalni Continue plati
    # jen pro tuto funkci.
    $ErrorActionPreference = "Continue"
    if ($env:SKIP_CUDA_CHECK -eq "1") {
        Write-Warn "SKIP_CUDA_CHECK=1 - preskakuji CUDA verifikaci."
        return
    }
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        Write-Warn "nvidia-smi nenalezen - nemuzu verifikovat CUDA verzi."
        return
    }
    $smiOutput = & nvidia-smi 2>$null | Out-String
    $match = [regex]::Match($smiOutput, "CUDA Version:\s*(\d+)\.(\d+)")
    if (-not $match.Success) { return }
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
            Write-Err "Detekovana CUDA $major.$minor - pipeline VYZADUJE CUDA 13+ pro NVFP4."
            Write-Err "Upgrade driveru >=580 a CUDA toolkit >=13.0."
        }
        throw "CUDA ${minMajor}+ required (detected $major.$minor)"
    }
    Write-Log "Preflight OK - CUDA $major.$minor detekovana (>=13.0)."
}

function Test-DataDir {
    if (-not (Test-Path "data")) {
        Write-Err "Chybi slozka data/ - nakopiruj tam vstupni JSON soubory"
        throw "data/ missing"
    }
    $jsonCount = (Get-ChildItem -Path "data" -Filter "*.json" -File | Measure-Object).Count
    if ($jsonCount -eq 0) {
        Write-Err "V data/ nenalezeny zadne *.json soubory"
        throw "no JSON files"
    }
    Write-Log "Detekovano $jsonCount JSON soubor(u) v data/"
}

# ---------------------------------------------------------------------------
# Per-phase smoke runner
# ---------------------------------------------------------------------------
function Invoke-ComposeBuild {
    param([string]$Profile, [string]$Service)
    # PS 5.1: redirect stderr nativniho prikazu (2>$null / 2>&1) bali radky
    # do ErrorRecordu; s globalnim $ErrorActionPreference=Stop by PRVNI progress
    # hlaska dockeru (jde na stderr) shodila skript. Lokalni Continue plati
    # jen pro tuto funkci.
    $ErrorActionPreference = "Continue"
    # Explicitni VIDITELNY build: `docker compose run` sam od sebe nerebuiluje
    # existujici image (jel by stary kod) a pri prvnim behu by build probihal
    # skryte na pozadi (stderr je u run -d potlaceny) - vypadalo by to zamrzle.
    Write-Log "Build image $Service (prvni beh stahuje zavislosti - muze trvat par minut)"
    docker compose --profile $Profile build $Service
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Build image $Service selhal (rc=$LASTEXITCODE)."
        return $false
    }
    return $true
}

function Invoke-PhaseViaProfile {
    param(
        [string]$Phase,
        [string]$Profile,
        [string]$LabelerService
    )
    # PS 5.1: redirect stderr nativniho prikazu (2>$null / 2>&1) bali radky
    # do ErrorRecordu; s globalnim $ErrorActionPreference=Stop by PRVNI progress
    # hlaska dockeru (jde na stderr) shodila skript. Lokalni Continue plati
    # jen pro tuto funkci.
    $ErrorActionPreference = "Continue"
    if (-not (Invoke-ComposeBuild -Profile $Profile -Service $LabelerService)) { return $false }
    Write-Log "Smoke FAZE [$Phase] - profile=$Profile (detached -d na pozadi)"
    # docker compose run -d prebehne up LLM dependency + spusti override command
    # v detached modu. Vrati container ID, ktery pak cekame pres docker wait.
    $cid = (docker compose --profile $Profile run -d --rm `
        -e SMOKE_CHECKPOINT_DIR=/app/checkpoints/smoke `
        $LabelerService `
        smoke-test --phase $Phase `
        --hw-label "$HwLabel" `
        --samples-per-dataset $SamplesPerDataset `
        --docker-startup-s $DockerStartupS 2>$null | Select-Object -Last 1)
    if (-not $cid) {
        Write-Err "[$Phase] docker compose run -d selhal na profilu $Profile"
        docker compose --profile $Profile down 2>&1 | Out-Null
        return $false
    }
    Write-Log "Container $cid bezi na pozadi. Sleduj: docker logs -f $cid"
    docker wait $cid 2>&1 | Out-Null
    $rc = [int](docker inspect --format='{{.State.ExitCode}}' $cid 2>$null)
    docker compose --profile $Profile down 2>&1 | Out-Null
    if ($rc -eq 0) {
        Write-Log "[$Phase] hotov pres $Profile"
        return $true
    }
    Write-Err "[$Phase] selhal na profilu $Profile (rc=$rc)"
    return $false
}

# ---------------------------------------------------------------------------
# NATIVE runtime helpers - bez dockeru, Python proti bezicimu LLM serveru
# ---------------------------------------------------------------------------
# $NativeAutoLlm: 1 (default) = auto-install + auto-start/stop SGLang
#                 0 = manual (user si LLM zapne sam)
$NativeAutoLlm = if ($env:NATIVE_AUTO_LLM) { $env:NATIVE_AUTO_LLM } else { "1" }

# Setup env vars 1:1 jako Docker (parita s docker-compose.yml LLM service)
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
    Write-Log "Native env parita: HF_XET=1, PYTHONWARNINGS=suppress,"
    Write-Log "  EXTRACTOR_BASE_URL=$($env:EXTRACTOR_BASE_URL)"
    Write-Log "  CRITIC_BASE_URL=$($env:CRITIC_BASE_URL)"
}

# Sestavi SGLang args 1:1 jako docker-compose.yml command block.
function Get-SglangArgsForPhase {
    param([string]$Phase)
    $args = @()
    switch ($Phase) {
        { $_ -in "taxonomy", "extract", "revise" } {
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
        Write-Log "LLM server na portu $Port uz bezi - pouzivam ho"
        return $true
    }
    if ($NativeAutoLlm -ne "1") {
        Write-Err "Endpoint http://localhost:$Port/v1 NEODPOVIDA. Manual mode: spust sam:"
        Write-Err "  .\scripts\start_native_llm.ps1 <model> $Port $GpuCount"
        return $false
    }
    $model = ""
    $tpSize = 1
    switch ($Phase) {
        { $_ -in "taxonomy", "extract", "revise" } {
            $model = if ($env:EXTRACTOR_MODEL) { $env:EXTRACTOR_MODEL } else { "RedHatAI/Qwen3.5-122B-A10B-NVFP4" }
            $tpSize = $GpuCount
        }
        "critic" {
            $model = if ($env:CRITIC_MODEL) { $env:CRITIC_MODEL } else { "nvidia/Gemma-4-31B-IT-NVFP4" }
            $tpSize = if ($env:CRITIC_TP_SIZE) { [int]$env:CRITIC_TP_SIZE } else { 1 }
        }
        default { return $true }
    }
    # Args 1:1 jako Docker
    $extra = Get-SglangArgsForPhase -Phase $Phase
    Write-Log "Auto-start SGLang pro '$Phase': $model na portu $Port (tp=$tpSize)"
    Write-Log "  Args: $extra"
    & "$PSScriptRoot\start_native_llm.ps1" -ModelPath $model -Port $Port -TpSize $tpSize -ExtraArgs $extra
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

# Cleanup hook - pri exit (Ctrl+C, fail, normal) zastav VSECHNY SGLang
# servery, ktere jsme my spustili. Iteruje pres PID files misto hardcoded
# ports - robustni pro budouci rozsireni.
function Invoke-NativeCleanup {
    if ($Runtime -ne "native" -or $NativeAutoLlm -ne "1") { return }
    $LogDir = if ($env:SGLANG_LOG_DIR) { $env:SGLANG_LOG_DIR } else { $env:TEMP }
    $pidFiles = Get-ChildItem -Path $LogDir -Filter "sglang_*.pid" -ErrorAction SilentlyContinue
    foreach ($pf in $pidFiles) {
        $port = $pf.BaseName -replace '^sglang_', ''
        Write-Warn "Cleanup: stop SGLang na portu $port (script exit)"
        try {
            & "$PSScriptRoot\stop_native_llm.ps1" -Port ([int]$port) 2>&1 | Out-Null
        } catch {}
    }
}
# PowerShell ekvivalent bash trap EXIT - exit handler pres engine event
Register-EngineEvent PowerShell.Exiting -Action { Invoke-NativeCleanup } | Out-Null

function Invoke-PhaseNative {
    param([string]$Phase)
    Write-Log "Smoke FAZE [$Phase] - NATIVE (Python primo, zadny docker)"

    if ($Phase -ne "finalize" -and $Phase -ne "report") {
        if (-not (Install-SgLangIfMissing)) { return $false }
        $port = switch ($Phase) {
            "taxonomy" { 8000 }
            "extract"  { 8000 }
            "critic"   { 8001 }
            "revise"   { 8000 }
        }
        if (-not (Start-NativeLlmForPhase -Phase $Phase -Port $port)) { return $false }
    }

    python -m labeler.main smoke-test --phase $Phase `
        --hw-label "$HwLabel" `
        --samples-per-dataset $SamplesPerDataset `
        --docker-startup-s $DockerStartupS
    $rc = $LASTEXITCODE

    if ($Phase -ne "finalize" -and $Phase -ne "report") {
        $port = switch ($Phase) { "taxonomy" { 8000 } "extract" { 8000 } "critic" { 8001 } "revise" { 8000 } }
        Stop-NativeLlmForPhase -Port $port
    }

    if ($rc -eq 0) {
        Write-Log "[$Phase] hotov (native)"
        return $true
    }
    Write-Err "[$Phase] selhal (native, exit $rc)"
    return $false
}

function Invoke-SmokeSample {
    Write-Log "FAZE 0a (smoke) - SAMPLE (vyber ukazek pro taxonomii, embeddingy bge-m3 na vLLM)"
    if ($Runtime -eq "native") {
    # Embedding server faze 0a bezi na vLLM, ktery na Windows nativne nebezi
    # (jen Linux) -> v native rezimu se faze preskoci; taxonomie pouzije
    # nahodny vyber. Pro embeddingy pust Docker runtime (nebo WSL + run_*.sh).
        Write-Warn "Native runtime: vLLM na Windows nebezi - faze sample preskocena, taxonomie pouzije nahodny vyber."
        return $true
    }
    $ok = $false
    try {
        $ok = Invoke-PhaseViaProfile -Phase "sample" -Profile "embed-vllm" -LabelerService "labeler-embed-vllm"
    } catch { $ok = $false }
    if (-not $ok) {
        Write-Warn "Vyber ukazek pres embeddingy selhal - taxonomie pouzije nahodny vyber."
    }
    return $true
}

function Invoke-SmokeTaxonomy {
    Write-Log "FAZE 0 (smoke) - TAXONOMY (auto-discovery + embeddingy)"
    if ($Runtime -eq "native") {
        if (-not (Invoke-PhaseNative -Phase "taxonomy")) { throw "taxonomy failed (native)" }
        return
    }
    # Taxonomy pouziva extractor LLM -> bezi pres extractor profily
    foreach ($backend in $ExtractorBackends) {
        if (Invoke-PhaseViaProfile -Phase "taxonomy" -Profile "extract-$backend" -LabelerService "labeler-extract-$backend") {
            return $true
        }
        Write-Warn "Backend $backend selhal pro taxonomy, zkousim dalsi..."
    }
    Write-Err "Vsechny taxonomy backendy selhaly."
    throw "taxonomy failed"
}

function Invoke-SmokeExtract {
    Write-Log "FAZE 1 (smoke) - EXTRACT"
    if ($Runtime -eq "native") {
        if (-not (Invoke-PhaseNative -Phase "extract")) { throw "extract failed (native)" }
        return
    }
    foreach ($backend in $ExtractorBackends) {
        if (Invoke-PhaseViaProfile -Phase "extract" -Profile "extract-$backend" -LabelerService "labeler-extract-$backend") {
            return $true
        }
        Write-Warn "Backend $backend selhal pro extract, zkousim dalsi…"
    }
    Write-Err "Vsechny extract backendy selhaly."
    throw "extract failed"
}

function Invoke-SmokeCritic {
    Write-Log "FAZE 2 (smoke) - CRITIC"
    if ($Runtime -eq "native") {
        if (-not (Invoke-PhaseNative -Phase "critic")) { throw "critic failed (native)" }
        return
    }
    foreach ($backend in $CriticBackends) {
        if (Invoke-PhaseViaProfile -Phase "critic" -Profile "critic-$backend" -LabelerService "labeler-critic-$backend") {
            return $true
        }
        Write-Warn "Backend $backend selhal pro critic, zkousim dalsi…"
    }
    Write-Err "Vsechny critic backendy selhaly."
    throw "critic failed"
}

function Invoke-SmokeRevise {
    Write-Log "FAZE 2b (smoke) - REVISE"
    if ($Runtime -eq "native") {
        if (-not (Invoke-PhaseNative -Phase "revise")) { throw "revise failed (native)" }
        return
    }
    foreach ($backend in $ReviseBackends) {
        if (Invoke-PhaseViaProfile -Phase "revise" -Profile "revise-$backend" -LabelerService "labeler-revise-$backend") {
            return $true
        }
        Write-Warn "Backend $backend selhal pro revise, zkousim dalsi…"
    }
    Write-Err "Vsechny revise backendy selhaly."
    throw "revise failed"
}

function Invoke-SmokeFinalize {
    # PS 5.1: redirect stderr nativniho prikazu (2>$null / 2>&1) bali radky
    # do ErrorRecordu; s globalnim $ErrorActionPreference=Stop by PRVNI progress
    # hlaska dockeru (jde na stderr) shodila skript. Lokalni Continue plati
    # jen pro tuto funkci.
    $ErrorActionPreference = "Continue"
    Write-Log "FAZE 3 (smoke) - FINALIZE (bez LLM, detached -d)"
    if ($Runtime -eq "native") {
        if (-not (Invoke-PhaseNative -Phase "finalize")) { throw "finalize failed (native)" }
        return
    }
    if (-not (Invoke-ComposeBuild -Profile "finalize" -Service "labeler-finalize")) { throw "finalize failed" }
    $cid = (docker compose --profile finalize run -d --rm `
        -e SMOKE_CHECKPOINT_DIR=/app/checkpoints/smoke `
        labeler-finalize `
        smoke-test --phase finalize `
        --hw-label "$HwLabel" `
        --samples-per-dataset $SamplesPerDataset `
        --docker-startup-s $DockerStartupS 2>$null | Select-Object -Last 1)
    if (-not $cid) {
        Write-Err "Finalize smoke: docker compose run -d selhal"
        throw "finalize failed"
    }
    Write-Log "Container $cid bezi na pozadi. Sleduj: docker logs -f $cid"
    docker wait $cid 2>&1 | Out-Null
    $rc = [int](docker inspect --format='{{.State.ExitCode}}' $cid 2>$null)
    if ($rc -ne 0) {
        Write-Err "Finalize smoke selhal (rc=$rc)"
        throw "finalize failed"
    }
    Write-Log "Finalize smoke hotov"
}

function Invoke-SmokeReport {
    # PS 5.1: redirect stderr nativniho prikazu (2>$null / 2>&1) bali radky
    # do ErrorRecordu; s globalnim $ErrorActionPreference=Stop by PRVNI progress
    # hlaska dockeru (jde na stderr) shodila skript. Lokalni Continue plati
    # jen pro tuto funkci.
    $ErrorActionPreference = "Continue"
    Write-Log "FAZE 4 (smoke) - REPORT (extrapolace + +/-2h konfidence)"
    if ($Runtime -eq "native") {
        if (-not (Invoke-PhaseNative -Phase "report")) { throw "report failed (native)" }
        return
    }
    if (-not (Invoke-ComposeBuild -Profile "finalize" -Service "labeler-finalize")) { throw "report failed" }
    $cid = (docker compose --profile finalize run -d --rm `
        -e SMOKE_CHECKPOINT_DIR=/app/checkpoints/smoke `
        labeler-finalize `
        smoke-test --phase report `
        --hw-label "$HwLabel" `
        --samples-per-dataset $SamplesPerDataset `
        --docker-startup-s $DockerStartupS 2>$null | Select-Object -Last 1)
    if (-not $cid) {
        Write-Err "Report: docker compose run -d selhal"
        throw "report failed"
    }
    docker wait $cid 2>&1 | Out-Null
    docker logs $cid 2>$null
    $rc = [int](docker inspect --format='{{.State.ExitCode}}' $cid 2>$null)
    if ($rc -ne 0) { Write-Err "Report selhal (rc=$rc)"; throw "report failed" }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
Test-CudaPreflight
Test-DataDir

New-Item -ItemType Directory -Force -Path "checkpoints/smoke" | Out-Null

Write-Log "Smoke test config:"
Write-Host "  Runtime=$Runtime"
Write-Host "  GpuType=$GpuType"
Write-Host "  GpuCount=$($GpuCount)x $gpuLabel"
Write-Host "  EXTRACTOR_TP_SIZE=$($env:EXTRACTOR_TP_SIZE)"
Write-Host "  CRITIC_TP_SIZE=$($env:CRITIC_TP_SIZE)"
Write-Host "  HwLabel=$HwLabel"
Write-Host "  SamplesPerDataset=$SamplesPerDataset"
Write-Host "  TAXONOMY_SAMPLES_PER_DATASET=$($env:TAXONOMY_SAMPLES_PER_DATASET)"
Write-Host "  SmokePhases=$($SmokePhases -join ',')"
Write-Host "  DockerStartupS=$DockerStartupS"
if ($Runtime -eq "native") {
    $extUrl = if ($env:EXTRACTOR_BASE_URL) { $env:EXTRACTOR_BASE_URL } else { "http://localhost:8000/v1" }
    $critUrl = if ($env:CRITIC_BASE_URL)   { $env:CRITIC_BASE_URL }   else { "http://localhost:8001/v1" }
    Write-Host "  EXTRACTOR_BASE_URL=$extUrl (musi uz bezet)"
    Write-Host "  CRITIC_BASE_URL=$critUrl (musi uz bezet)"
}
Write-Host "  Output: checkpoints/smoke/"

# V native modu chceme aby smoke_test.py psal do checkpoints/smoke/
if ($Runtime -eq "native") {
    if (-not $env:SMOKE_CHECKPOINT_DIR) {
        $env:SMOKE_CHECKPOINT_DIR = Join-Path (Get-Location) "checkpoints/smoke"
    }
}

foreach ($phase in $SmokePhases) {
    switch ($phase) {
        "sample"   { Invoke-SmokeSample   | Out-Null }
        "taxonomy" { Invoke-SmokeTaxonomy | Out-Null }
        "extract"  { Invoke-SmokeExtract  | Out-Null }
        "critic"   { Invoke-SmokeCritic   | Out-Null }
        "revise"   { Invoke-SmokeRevise   | Out-Null }
        "finalize" { Invoke-SmokeFinalize }
        default {
            Write-Err "Neznama faze: $phase"
            throw "unknown phase"
        }
    }
}

Invoke-SmokeReport

Write-Log "Smoke test hotov."
Write-Host ""
Write-Host "Artefakty:"
Write-Host "  checkpoints/smoke/smoke_timings.json     - raw per-fazova mereni"
Write-Host "  checkpoints/smoke/smoke_test_report.md   - human-readable report"
Write-Host "  checkpoints/smoke/smoke_test_report.json - strojovy export"
