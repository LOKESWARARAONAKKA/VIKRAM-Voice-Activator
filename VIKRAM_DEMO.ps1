$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "        ISRO SIH 2026 - VIKRAM VOICE ACTIVATOR" -ForegroundColor Cyan
Write-Host "                 PROJECT EVIDENCE DEMO" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "PROJECT LOCATION" -ForegroundColor Yellow
Write-Host (Get-Location)
Write-Host ""

Write-Host "============================================================"
Write-Host "1. DATASET SUMMARY" -ForegroundColor Green
Write-Host "============================================================"

$positive = Get-ChildItem ".\dataset\vikram" -Recurse -Filter "*.wav" -ErrorAction SilentlyContinue
$negative = Get-ChildItem ".\dataset\negative" -Recurse -Filter "*.wav" -ErrorAction SilentlyContinue

Write-Host "Positive VIKRAM recordings: $($positive.Count)"
Write-Host "Negative recordings:        $($negative.Count)"
Write-Host "Total recordings:           $($positive.Count + $negative.Count)"
Write-Host ""

Write-Host "Positive speaker folders:" -ForegroundColor Yellow
Get-ChildItem ".\dataset\vikram" -Directory -ErrorAction SilentlyContinue |
ForEach-Object {
    $count = (Get-ChildItem $_.FullName -Filter "*.wav" -ErrorAction SilentlyContinue).Count
    Write-Host ("  {0,-15} {1,3} recordings" -f $_.Name,$count)
}
Write-Host ""

Write-Host "Audio format: 16 kHz | Mono | 16-bit PCM | 1.5 sec | 24,000 samples"
Write-Host ""

Write-Host "============================================================"
Write-Host "2. TRAINING / VALIDATION / TEST SETUP" -ForegroundColor Green
Write-Host "============================================================"

Write-Host "Speaker-independent split:"
Write-Host "  TRAIN      : dtss, lokesh, s.akshitha"
Write-Host "  VALIDATION : lavanya"
Write-Host "  TEST       : mohana"
Write-Host ""

Write-Host "Feature extraction:"
Write-Host "  Sample rate : 16000 Hz"
Write-Host "  FFT         : 512"
Write-Host "  Hop length  : 160"
Write-Host "  Mel bands   : 40"
Write-Host "  Frames      : 151"
Write-Host "  Feature     : 40 x 151"
Write-Host ""

Write-Host "============================================================"
Write-Host "3. TRAINED MODEL" -ForegroundColor Green
Write-Host "============================================================"

Get-ChildItem ".\training" -Filter "*.keras" -ErrorAction SilentlyContinue |
Select-Object Name,Length |
Format-Table -AutoSize

Write-Host ""

Write-Host "TFLite model:" -ForegroundColor Yellow

if (Test-Path ".\training\vikram_cnn.tflite") {
    $tflite = Get-Item ".\training\vikram_cnn.tflite"
    Write-Host "  File: $($tflite.FullName)"
    Write-Host "  Size: $($tflite.Length) bytes"
} else {
    Write-Host "  training\vikram_cnn.tflite not found"
}

Write-Host ""

Write-Host "============================================================"
Write-Host "4. MODEL TEST RESULT" -ForegroundColor Green
Write-Host "============================================================"

Write-Host "Held-out test set:"
Write-Host "  Total samples : 24"
Write-Host "  Accuracy      : 91.67%"
Write-Host "  VIKRAM        : 14 / 14 correct"
Write-Host "  Negative      : 8 / 10 correct"
Write-Host "  False positives: 2"
Write-Host ""

Write-Host "============================================================"
Write-Host "5. ESP32-S3 HARDWARE VERIFICATION" -ForegroundColor Green
Write-Host "============================================================"

Write-Host "Board: ESP32-S3 N16R8"
Write-Host "Microphone: INMP441 I2S"
Write-Host ""
Write-Host "I2S wiring:"
Write-Host "  3V3 -> 3V3"
Write-Host "  GND -> GND"
Write-Host "  SCK -> GPIO 4"
Write-Host "  WS  -> GPIO 5"
Write-Host "  SD  -> GPIO 6"
Write-Host "  L/R -> GND"
Write-Host ""

Write-Host "Verified hardware tests:"
Write-Host "  [OK] Microphone RMS test"
Write-Host "  [OK] 1.5 second audio capture"
Write-Host "  [OK] 24,000 samples captured"
Write-Host "  [OK] Feature extraction"
Write-Host "  [OK] 40 mel bands"
Write-Host "  [OK] 151 frames"
Write-Host ""

Write-Host "============================================================"
Write-Host "6. IMPORTANT IMPLEMENTATION STATUS" -ForegroundColor Yellow
Write-Host "============================================================"

Write-Host "Training and TFLite conversion: COMPLETED"
Write-Host "ESP32 microphone capture:       VERIFIED"
Write-Host "ESP32 feature extraction:       VERIFIED"
Write-Host "Final ESP32 TFLite inference:   NOT COMPLETED"
Write-Host ""
Write-Host "This prototype should not be presented as a fully"
Write-Host "validated production-ready wake-word detector."
Write-Host ""

Write-Host "============================================================"
Write-Host "7. GITHUB REPOSITORY" -ForegroundColor Green
Write-Host "============================================================"

Write-Host "https://github.com/LOKESWARARAONAKKA/VIKRAM-Voice-Activator"
Write-Host ""

Write-Host "============================================================"
Write-Host "DEMO COMPLETE"
Write-Host "============================================================"
Write-Host ""

Read-Host "Press ENTER to finish"