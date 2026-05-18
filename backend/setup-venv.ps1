# Python 3.12 가상환경 생성 및 의존성 설치
# 사용: .\setup-venv.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (py -3.12 --version 2>$null)) {
    Write-Host "Python 3.12 없음. 설치: py install 3.12"
    exit 1
}

if (Test-Path .venv) {
    Write-Host "기존 .venv 삭제 중..."
    Remove-Item -Recurse -Force .venv
}

Write-Host "Python 3.12 venv 생성..."
py -3.12 -m venv .venv

Write-Host "패키지 설치 (수 분 소요)..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
Write-Host "PaddleOCR 2.7 설치 (3.x 제거 후)..."
.\.venv\Scripts\pip.exe uninstall paddlepaddle paddleocr paddlex -y 2>$null
.\.venv\Scripts\pip.exe install -r requirements-paddle.txt

Write-Host ""
Write-Host "완료. 실행:"
Write-Host "  .\.venv\Scripts\activate"
Write-Host "  uvicorn app.main:app --reload --port 8000"
