param(
    [string]$EndpointId = "ex9647vtulowka",
    [int]$LocalPort = 9000,
    [int]$WarmTimeoutSeconds = 300,
    [int]$RetryDelaySeconds = 10
)

$ErrorActionPreference = "Stop"
$proxyProcess = $null
$secret = $null
$apiKey = $null

function Get-HttpStatusCode {
    param([System.Management.Automation.ErrorRecord]$ErrorRecord)

    try {
        if ($null -ne $ErrorRecord.Exception.Response.StatusCode) {
            return [int]$ErrorRecord.Exception.Response.StatusCode
        }
    }
    catch {
        return $null
    }
    return $null
}

function Get-ErrorDetail {
    param([System.Management.Automation.ErrorRecord]$ErrorRecord)

    if ($null -ne $ErrorRecord.ErrorDetails -and $ErrorRecord.ErrorDetails.Message) {
        return $ErrorRecord.ErrorDetails.Message
    }
    return $ErrorRecord.Exception.Message
}

try {
    Write-Host "Checking the local proxy dependency..."
    python.exe -c "import aiohttp" 2>$null
    if ($LASTEXITCODE -ne 0) {
        python.exe -m pip install --user aiohttp
        if ($LASTEXITCODE -ne 0) {
            throw "Could not install aiohttp for the local proxy."
        }
    }

    $secret = Read-Host "Paste your RunPod API key" -AsSecureString
    $apiKey = [System.Net.NetworkCredential]::new("", $secret).Password
    $headers = @{ Authorization = "Bearer $apiKey" }
    $baseUrl = "https://$EndpointId.api.runpod.ai"

    Write-Host "Starting the JoyAI worker through GET /health..."
    Write-Host "The H200 must load the 31 GB DiT checkpoint before RunPod routes traffic."
    Write-Host "A RunPod 400 'timed out waiting for worker' response is retried until the $WarmTimeoutSeconds-second safety limit."

    $warmTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $health = $null
    $workerReady = $false
    $attempt = 0

    while (-not $workerReady -and $warmTimer.Elapsed.TotalSeconds -lt $WarmTimeoutSeconds) {
        $attempt += 1
        $remainingSeconds = $WarmTimeoutSeconds - [int]$warmTimer.Elapsed.TotalSeconds
        if ($remainingSeconds -le 0) {
            break
        }

        # RunPod's load balancer can spend about two minutes waiting for a cold
        # worker before returning HTTP 400. Keep each request slightly above
        # that window, then retry without creating another worker.
        $requestTimeout = [Math]::Min(135, $remainingSeconds)
        Write-Host "Readiness attempt $attempt (elapsed $([int]$warmTimer.Elapsed.TotalSeconds)s)..."

        try {
            $health = Invoke-RestMethod `
                -Uri "$baseUrl/health" `
                -Headers $headers `
                -Method Get `
                -TimeoutSec $requestTimeout

            if ($health.ok -and $health.runtime_loaded) {
                $workerReady = $true
                break
            }

            Write-Host "Worker answered, but the runtime is still initializing."
        }
        catch {
            $statusCode = Get-HttpStatusCode $_
            $detail = Get-ErrorDetail $_
            $retryableStatusCodes = @(400, 408, 425, 429, 500, 502, 503, 504)
            $isColdStartResponse = (
                $retryableStatusCodes -contains $statusCode -or
                $detail -match "timed out waiting for worker|cold start|no worker|temporarily unavailable|semaphore timeout|forcibly closed"
            )

            if (-not $isColdStartResponse) {
                throw
            }

            Write-Host "Worker is still cold-starting: $detail"
        }

        $remainingSeconds = $WarmTimeoutSeconds - [int]$warmTimer.Elapsed.TotalSeconds
        if ($remainingSeconds -gt 0) {
            $sleepSeconds = [Math]::Min($RetryDelaySeconds, $remainingSeconds)
            Write-Host "Retrying in $sleepSeconds seconds..."
            Start-Sleep -Seconds $sleepSeconds
        }
    }

    $warmTimer.Stop()
    if (-not $workerReady) {
        throw "JoyAI did not become ready within $WarmTimeoutSeconds seconds. Stop here and inspect the worker log; do not rebuild the image or start another Pod."
    }

    Write-Host "JoyAI runtime is ready after $([int]$warmTimer.Elapsed.TotalSeconds) seconds."

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

    $localUrl = "http://127.0.0.1:$LocalPort/"
    $localHealthUrl = "${localUrl}health"
    $localReady = $false
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        if ($proxyProcess.HasExited) {
            throw "The local JoyAI proxy stopped unexpectedly."
        }
        try {
            $localHealth = Invoke-RestMethod `
                -Uri $localHealthUrl `
                -Method Get `
                -TimeoutSec 30
            if ($localHealth.ok -and $localHealth.runtime_loaded) {
                $localReady = $true
                break
            }
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    if (-not $localReady) {
        throw "The local proxy could not reach the ready JoyAI worker."
    }

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
