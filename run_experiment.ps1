# PowerShell script for running BERT fine-tuning experiments on Windows
# Usage: .\run_experiment.ps1 -ExperimentType "basic" or "advanced" or "custom"

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("basic", "advanced", "multiclass", "small", "test", "custom")]
    [string]$ExperimentType,
    
    [string]$DatasetName = "imdb",
    [string]$ModelName = "bert-base-uncased",
    [string]$OutputDir = "",
    [int]$Epochs = 3,
    [int]$BatchSize = 16,
    [double]$LearningRate = 0.00002,
    [int]$MaxSamples = 0,
    [switch]$UseWandb,
    [string]$WandbProject = "bert-finetuning",
    [switch]$UseFP16,
    [switch]$Help
)

function Show-Help {
    Write-Host @"
BERT Fine-tuning Experiment Runner for Windows

Usage: .\run_experiment.ps1 -ExperimentType <type> [options]

Experiment Types:
  basic      - Basic IMDB sentiment analysis training
  advanced   - Advanced training with optimal settings
  multiclass - Multi-class classification (AG News)
  small      - Small dataset/quick test
  test       - Setup validation test
  custom     - Custom training with specified parameters

Options:
  -DatasetName     Dataset to use (default: imdb)
  -ModelName       Model to use (default: bert-base-uncased)
  -OutputDir       Output directory (auto-generated if not specified)
  -Epochs          Number of training epochs (default: 3)
  -BatchSize       Batch size per device (default: 16)
  -LearningRate    Learning rate (default: 0.00002)
  -MaxSamples      Maximum samples to use (0 = all)
  -UseWandb        Enable Weights & Biases tracking
  -WandbProject    W&B project name (default: bert-finetuning)
  -UseFP16         Enable mixed precision training
  -Help            Show this help message

Examples:
  .\run_experiment.ps1 -ExperimentType basic
  .\run_experiment.ps1 -ExperimentType advanced -UseWandb -UseFP16
  .\run_experiment.ps1 -ExperimentType custom -DatasetName ag_news -Epochs 5
  .\run_experiment.ps1 -ExperimentType test
"@
}

function Test-PythonSetup {
    Write-Host "Checking Python setup..." -ForegroundColor Yellow
    
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Python not found. Please install Python 3.7+" -ForegroundColor Red
        return $false
    }
    
    # Check if requirements are installed
    $packages = @("torch", "transformers", "datasets", "sklearn")
    foreach ($package in $packages) {
        try {
            python -c "import $package; print('✓ $package')" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ $package installed" -ForegroundColor Green
            } else {
                Write-Host "✗ $package not installed" -ForegroundColor Red
                Write-Host "  Run: pip install -r requirements.txt" -ForegroundColor Yellow
                return $false
            }
        }
        catch {
            Write-Host "✗ $package not installed" -ForegroundColor Red
            return $false
        }
    }
    
    return $true
}

function Start-BasicExperiment {
    Write-Host "Starting basic IMDB sentiment analysis..." -ForegroundColor Cyan
    
    $outputDir = if ($OutputDir) { $OutputDir } else { ".\results\imdb_basic_$(Get-Date -Format 'yyyyMMdd_HHmmss')" }
    
    $args = @(
        "train.py",
        "--dataset_name", "imdb",
        "--model_name", "bert-base-uncased",
        "--output_dir", $outputDir,
        "--num_train_epochs", "3",
        "--per_device_train_batch_size", "16",
        "--learning_rate", "2e-5",
        "--evaluation_strategy", "epoch",
        "--save_strategy", "epoch",
        "--logging_steps", "100",
        "--seed", "42"
    )
    
    if ($UseWandb) {
        $args += @("--report_to", "wandb", "--wandb_project", $WandbProject)
    } else {
        $args += @("--report_to", "tensorboard")
    }
    
    if ($UseFP16) {
        $args += @("--fp16", "true")
    }
    
    Write-Host "Command: python $($args -join ' ')" -ForegroundColor Gray
    python @args
}

function Start-AdvancedExperiment {
    Write-Host "Starting advanced IMDB training..." -ForegroundColor Cyan
    
    $outputDir = if ($OutputDir) { $OutputDir } else { ".\results\imdb_advanced_$(Get-Date -Format 'yyyyMMdd_HHmmss')" }
    
    $args = @(
        "train.py",
        "--dataset_name", "imdb",
        "--model_name", "bert-large-uncased",
        "--output_dir", $outputDir,
        "--num_train_epochs", "5",
        "--per_device_train_batch_size", "8",
        "--learning_rate", "1e-5",
        "--warmup_ratio", "0.1",
        "--lr_scheduler_type", "cosine",
        "--early_stopping_patience", "3",
        "--evaluation_strategy", "steps",
        "--eval_steps", "500",
        "--save_strategy", "steps",
        "--save_steps", "500",
        "--logging_steps", "50",
        "--metric_for_best_model", "eval_f1",
        "--seed", "42"
    )
    
    if ($UseWandb) {
        $args += @("--report_to", "wandb", "--wandb_project", $WandbProject)
    } else {
        $args += @("--report_to", "tensorboard")
    }
    
    if ($UseFP16) {
        $args += @("--fp16", "true")
    }
    
    Write-Host "Command: python $($args -join ' ')" -ForegroundColor Gray
    python @args
}

function Start-MulticlassExperiment {
    Write-Host "Starting multi-class AG News training..." -ForegroundColor Cyan
    
    $outputDir = if ($OutputDir) { $OutputDir } else { ".\results\agnews_$(Get-Date -Format 'yyyyMMdd_HHmmss')" }
    
    $args = @(
        "train.py",
        "--dataset_name", "ag_news",
        "--num_labels", "4",
        "--model_name", "bert-base-uncased",
        "--output_dir", $outputDir,
        "--num_train_epochs", "3",
        "--per_device_train_batch_size", "32",
        "--learning_rate", "3e-5",
        "--max_length", "256",
        "--evaluation_strategy", "epoch",
        "--save_strategy", "epoch",
        "--logging_steps", "100",
        "--seed", "42"
    )
    
    if ($UseWandb) {
        $args += @("--report_to", "wandb", "--wandb_project", $WandbProject)
    } else {
        $args += @("--report_to", "tensorboard")
    }
    
    if ($UseFP16) {
        $args += @("--fp16", "true")
    }
    
    Write-Host "Command: python $($args -join ' ')" -ForegroundColor Gray
    python @args
}

function Start-SmallExperiment {
    Write-Host "Starting small dataset test..." -ForegroundColor Cyan
    
    $outputDir = if ($OutputDir) { $OutputDir } else { ".\results\small_test_$(Get-Date -Format 'yyyyMMdd_HHmmss')" }
    
    $args = @(
        "train.py",
        "--dataset_name", "imdb",
        "--max_samples", "1000",
        "--model_name", "distilbert-base-uncased",
        "--output_dir", $outputDir,
        "--num_train_epochs", "5",
        "--per_device_train_batch_size", "16",
        "--learning_rate", "5e-5",
        "--early_stopping_patience", "2",
        "--evaluation_strategy", "epoch",
        "--save_strategy", "epoch",
        "--logging_steps", "50",
        "--seed", "42"
    )
    
    if ($UseWandb) {
        $args += @("--report_to", "wandb", "--wandb_project", $WandbProject)
    } else {
        $args += @("--report_to", "tensorboard")
    }
    
    Write-Host "Command: python $($args -join ' ')" -ForegroundColor Gray
    python @args
}

function Start-TestExperiment {
    Write-Host "Running setup validation test..." -ForegroundColor Cyan
    python test_setup.py
}

function Start-CustomExperiment {
    Write-Host "Starting custom experiment..." -ForegroundColor Cyan
    
    $outputDir = if ($OutputDir) { $OutputDir } else { ".\results\custom_$(Get-Date -Format 'yyyyMMdd_HHmmss')" }
    
    $args = @(
        "train.py",
        "--dataset_name", $DatasetName,
        "--model_name", $ModelName,
        "--output_dir", $outputDir,
        "--num_train_epochs", $Epochs.ToString(),
        "--per_device_train_batch_size", $BatchSize.ToString(),
        "--learning_rate", $LearningRate.ToString(),
        "--evaluation_strategy", "epoch",
        "--save_strategy", "epoch",
        "--logging_steps", "100",
        "--seed", "42"
    )
    
    if ($MaxSamples -gt 0) {
        $args += @("--max_samples", $MaxSamples.ToString())
    }
    
    if ($UseWandb) {
        $args += @("--report_to", "wandb", "--wandb_project", $WandbProject)
    } else {
        $args += @("--report_to", "tensorboard")
    }
    
    if ($UseFP16) {
        $args += @("--fp16", "true")
    }
    
    Write-Host "Command: python $($args -join ' ')" -ForegroundColor Gray
    python @args
}

# Main script execution
if ($Help) {
    Show-Help
    exit 0
}

Write-Host @"
================================================
BERT Fine-tuning Experiment Runner
================================================
"@ -ForegroundColor Cyan

# Test Python setup
if (-not (Test-PythonSetup)) {
    Write-Host "Setup validation failed. Please install required packages." -ForegroundColor Red
    exit 1
}

# Run the specified experiment
switch ($ExperimentType) {
    "basic" { Start-BasicExperiment }
    "advanced" { Start-AdvancedExperiment }
    "multiclass" { Start-MulticlassExperiment }
    "small" { Start-SmallExperiment }
    "test" { Start-TestExperiment }
    "custom" { Start-CustomExperiment }
}

Write-Host "`nExperiment completed!" -ForegroundColor Green
