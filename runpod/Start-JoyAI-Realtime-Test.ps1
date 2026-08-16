param(
    [string]$EndpointId = "ex9647vtulowka",
    [int]$LocalPort = 9000,
    [int]$WarmTimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"
$proxyProcess = $null
$secret = $null
$apiKey = $null

try {
    $secret = Read-Host "Paste your RunPod API key" -AsSecureString
    $apiKey = [System.Net.NetworkCredential]::new("", $secret).Password
    $headers = @{ Authorization = "Bearer $apiKey" }
    $baseUrl = "https://$EndpointId.api.runpod.ai"

    Write-Host "Warming the JoyAI model through POST /load..."
    $loadResponse = Invoke-RestMethod `
        -Uri "$baseUrl/load" `
        -Headers $headers `
        -Method Post `
        -ContentType "application/json" `
        -Body "{}" `
        -TimeoutSec $WarmTimeoutSeconds

    Write-Host "Checking the public JoyAI health route..."
    $health = Invoke-RestMethod `
        -Uri "$baseUrl/health" `
        -Headers $headers `
        -Method Get `
        -TimeoutSec 30

    if (-not $health.ok -or -not $health.runtime_loaded) {
        throw "JoyAI reported unhealthy or the model is not loaded."
    }

    python.exe -m pip install --user aiohttp

    $env:RUNPOD_API_KEY = $apiKey
    $proxyScript = Join-Path $PSScriptRoot "local_proxy.py"
    $proxyProcess = Start-Process `
        -FilePath "python.exe" `
        -ArgumentList @(
            ('"{0}"' -f $proxyScript),
            "--endpoint-id", $EndpointId,
            "--port", $LocalPort
        ) `
        -NoNewWindow `
        -PassThru

    Start-Sleep -Seconds 2
    $localUrl = "http://127.0.0.1:$LocalPort/"
    Write-Host "JoyAI is ready: $localUrl"
    Write-Host "Keep this window open. Press Ctrl+C when the test is finished."
    Start-Process $localUrl
    Wait-Process -Id $proxyProcess.Id
}
catch {
    Write-Host "TEST FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    if ($null -ne $proxyProcess -and -not $proxyProcess.HasExited) {
        Stop-Process -Id $proxyProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item Env:RUNPOD_API_KEY -ErrorAction SilentlyContinue
    $apiKey = $null
    $secret = $null
}
