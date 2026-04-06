#!/usr/bin/env python3
"""
Validator CLI commands for EvolAI subnet

Commands:
- setup: Verify validator environment
- get-miners: List registered miners from the Bittensor chain
- config: Manage validator configuration
- list-local: View local evaluation results
- run: Continuous loss-based evaluation loop with auto weight setting
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
console = Console()
err_console = Console(stderr=True, style="bold red")

# Validator CLI app
validator_app = typer.Typer(
    name="validator",
    help="Validator commands for running evaluations and submitting results",
    no_args_is_help=True
)

_NUM_QUESTIONS = 1  # keep in sync with validator.config.NUM_QUESTIONS


# ──────────────────────────────────────────────────────────────────────────────
# evolcli validator setup — preflight check
# ──────────────────────────────────────────────────────────────────────────────

@validator_app.command("setup")
def setup_check():
    """
    Verify that the validator environment is correctly configured.

    Checks:
    - Python version
    - Core packages (bittensor, torch, transformers)
    - Optional vLLM binary reachable (not required for current loss-based evaluation)
    - GPU availability & VRAM
    - .env file present
    """
    from dotenv import load_dotenv
    load_dotenv()
    import shutil
    import subprocess as _sp

    console.print("\n[bold cyan]EvolAI Validator — Environment Check[/bold cyan]\n")

    ok_count = 0
    warn_count = 0
    fail_count = 0

    def ok(msg):
        nonlocal ok_count; ok_count += 1
        console.print(f"  [green]✓[/green] {msg}")

    def warn(msg):
        nonlocal warn_count; warn_count += 1
        console.print(f"  [yellow]⚠[/yellow] {msg}")

    def fail(msg):
        nonlocal fail_count; fail_count += 1
        console.print(f"  [red]✗[/red] {msg}")

    # ── Python ────────────────────────────────────────────────────────────
    import sys
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 9):
        ok(f"Python {py_ver}")
    else:
        fail(f"Python {py_ver} — need >= 3.9")

    # ── Core packages ─────────────────────────────────────────────────────
    for pkg in ["bittensor", "torch", "transformers", "openai", "httpx", "rich"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{pkg} {ver}")
        except ImportError:
            fail(f"{pkg} not installed")

    # ── Optional vLLM binary ──────────────────────────────────────────────
    vllm_bin = os.environ.get("VLLM_EXECUTABLE", "vllm")
    vllm_path = shutil.which(vllm_bin)
    if vllm_path:
        try:
            result = _sp.run([vllm_path, "--version"], capture_output=True, text=True, timeout=10)
            ver = result.stdout.strip() or result.stderr.strip() or "unknown"
            ok(f"Optional vLLM binary: {vllm_path} ({ver})")
        except Exception:
            ok(f"Optional vLLM binary: {vllm_path}")
    elif vllm_bin != "vllm":
        warn(f"Optional vLLM binary not found at: {vllm_bin}")
    else:
        warn(
            "vLLM binary not on PATH. This is OK for the current HuggingFace "
            "loss evaluation flow."
        )

    # ── GPUs ──────────────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            gpus = []
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                gpus.append(f"GPU {i}: {name} ({mem:.0f} GB)")
            ok(f"{n} GPU(s) detected")
            for g in gpus:
                console.print(f"      {g}")

            if n >= 4:
                console.print(
                    "    [dim]Recommended: --parallel-miners 3 "
                    "(judge GPU 0, miners GPU 1-3)[/dim]"
                )
            elif n >= 2:
                console.print(
                    "    [dim]Recommended: default layout "
                    "(judge GPU 0, miner GPU 1)[/dim]"
                )
        else:
            warn("No CUDA GPUs detected — torch.cuda.is_available() = False")
    except ImportError:
        fail("torch not installed — cannot check GPUs")

    # ── .env file ─────────────────────────────────────────────────────────
    env_file = Path(".env")
    if env_file.exists():
        ok(f".env file present ({env_file.resolve()})")
    else:
        warn(".env file not found — run [dim]bash scripts/setup-validator.sh[/dim] to create one")

    # ── Summary ───────────────────────────────────────────────────────────
    console.print()
    parts = [f"[green]{ok_count} passed[/green]"]
    if warn_count:
        parts.append(f"[yellow]{warn_count} warnings[/yellow]")
    if fail_count:
        parts.append(f"[red]{fail_count} failed[/red]")
    console.print(f"  {', '.join(parts)}\n")

    if fail_count:
        raise typer.Exit(1)


@validator_app.command("get-miners")
def get_miners(
    track: Optional[str] = typer.Option(
        None,
        "--track",
        help="Filter by track: transformer, mamba2, or all"
    ),
    netuid: int = typer.Option(
        47,
        "--netuid",
        help="Subnet netuid (default: 47)"
    ),
    network: str = typer.Option(
        "finney",
        "--network",
        help="Bittensor network: finney, test, or local"
    )
):
    """
    Fetch registered miners directly from Bittensor chain commitment metadata

    Reads on-chain commitment data for every UID in the subnet metagraph,
    decodes the compressed model metadata, and displays a summary table.

    Shows:
    - Miner UID and hotkey
    - Registered models for each track (transformer / mamba2)
    - Model revisions
    """
    from dotenv import load_dotenv
    load_dotenv()
    from evolai.utils.metadata import decompress_metadata

    # Interactive prompt for track if not provided
    if track is None:
        track = Prompt.ask(
            "Select track",
            choices=["transformer", "mamba2", "all"],
            default="all"
        )

    # Normalize track input
    track = track.lower()
    if track in ["both", "all"]:
        track = None

    # Connect to Bittensor and load metagraph
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task(f"Connecting to Bittensor ({network})...", total=None)
        try:
            import bittensor as bt
            subtensor = bt.Subtensor(network=network)
            metagraph = subtensor.metagraph(netuid)
        except Exception as e:
            err_console.print(f"\n❌ Failed to connect to Bittensor: {e}")
            raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] Connected — subnet [bold]{netuid}[/bold], [bold]{len(metagraph.hotkeys)}[/bold] UIDs")

    miners = []
    uids_no_meta = []

    # Initialize HF API for upload timestamps
    try:
        from huggingface_hub import HfApi as _HfApi
        _hf_api = _HfApi()
    except Exception:
        _hf_api = None

    def _get_hf_upload_time(model_name, revision):
        if not _hf_api or not model_name or model_name == '-':
            return None
        try:
            repo_info = _hf_api.repo_info(model_name, revision=revision, repo_type="model")
            commits = list(_hf_api.list_repo_commits(model_name, repo_type="model"))
            revision_to_check = revision or repo_info.sha
            for commit in commits:
                if commit.commit_id == revision_to_check or commit.commit_id.startswith(revision_to_check or ""):
                    return commit.created_at
            return repo_info.last_modified
        except Exception:
            return None

    # Scan each UID's on-chain commitment
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Scanning chain commitments...", total=len(metagraph.hotkeys))

        for uid in range(len(metagraph.hotkeys)):
            hotkey = metagraph.hotkeys[uid]
            progress.update(task, advance=1, description=f"Scanning UID {uid}/{len(metagraph.hotkeys) - 1}…")

            try:
                commit_data = subtensor.get_commitment_metadata(netuid, hotkey)
                if not commit_data:
                    uids_no_meta.append(uid)
                    continue

                # Extract bytes from Bittensor's nested commitment structure:
                # commit_data['info']['fields'][0][0]['Raw50'][0] → tuple of ints → bytes
                if not (isinstance(commit_data, dict) and 'info' in commit_data):
                    uids_no_meta.append(uid)
                    continue

                fields = commit_data['info']['fields']
                if not (fields and len(fields) > 0 and fields[0] and len(fields[0]) > 0):
                    uids_no_meta.append(uid)
                    continue

                raw_data = fields[0][0]
                # Field key is RawN where N = byte length (e.g. Raw32, Raw50, Raw64)
                raw_key = next((k for k in raw_data if k.startswith('Raw') and k[3:].isdigit()), None)
                if raw_key is None:
                    uids_no_meta.append(uid)
                    continue

                compressed_bytes = bytes(raw_data[raw_key][0])
                metadata = decompress_metadata(compressed_bytes)
                if not metadata:
                    uids_no_meta.append(uid)
                    continue

                transformer_info = metadata.get('transformer', {})
                mamba2_info = metadata.get('mamba2', {})
                has_transformer = bool(transformer_info.get('model_name'))
                has_mamba2 = bool(mamba2_info.get('model_name'))

                # Apply track filter
                if track == 'transformer' and not has_transformer:
                    continue
                if track == 'mamba2' and not has_mamba2:
                    continue
                if not has_transformer and not has_mamba2:
                    uids_no_meta.append(uid)
                    continue

                miners.append({
                    'uid': uid,
                    'hotkey': hotkey,
                    'metadata': metadata,
                    't_upload': _get_hf_upload_time(transformer_info.get('model_name'), transformer_info.get('revision')) if has_transformer else None,
                    'm_upload': _get_hf_upload_time(mamba2_info.get('model_name'), mamba2_info.get('revision')) if has_mamba2 else None,
                })

            except Exception:
                uids_no_meta.append(uid)

    # Display result
    if not miners:
        console.print(f"\n[yellow]No miners registered{' for ' + track + ' track' if track else ''}[/yellow]\n")
        return

    table = Table(
        title=f"Registered Miners — Subnet {netuid}{' (' + track.upper() + ' track)' if track else ''}",
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("UID", style="dim", justify="right")
    table.add_column("Hotkey", style="bold")
    table.add_column("Transformer Model")
    table.add_column("T-Rev", style="dim")
    table.add_column("T-Upload", style="dim")
    table.add_column("Mamba2 Model")
    table.add_column("M-Rev", style="dim")
    table.add_column("M-Upload", style="dim")

    for miner in miners:
        uid = miner['uid']
        hotkey = miner['hotkey']
        meta = miner['metadata']

        transformer_info = meta.get('transformer', {})
        mamba2_info = meta.get('mamba2', {})

        t_model = transformer_info.get('model_name', '-')
        t_rev = str(transformer_info.get('revision', '-') or 'main')
        t_up = miner.get('t_upload')
        t_up_str = t_up.strftime('%Y-%m-%d') if t_up else '-'

        m_model = mamba2_info.get('model_name', '-')
        m_rev = str(mamba2_info.get('revision', '-') or 'main')
        m_up = miner.get('m_upload')
        m_up_str = m_up.strftime('%Y-%m-%d') if m_up else '-'

        table.add_row(
            str(uid),
            hotkey[:16] + "…" if len(hotkey) > 16 else hotkey,
            t_model,
            t_rev[:10] + "…" if len(t_rev) > 10 else t_rev,
            t_up_str,
            m_model,
            m_rev[:10] + "…" if len(m_rev) > 10 else m_rev,
            m_up_str,
        )

    console.print("\n")
    console.print(table)
    console.print(f"\n[dim]Total: {len(miners)} miners | UIDs without metadata: {len(uids_no_meta)}[/dim]\n")


@validator_app.command("config")
def validator_config(
    show: Optional[bool] = typer.Option(
        False,
        "--show",
        help="Show current configuration"
    ),
    set_key: Optional[str] = typer.Option(
        None,
        "--set",
        help="Set configuration key (format: key=value)"
    )
):
    """
    Show or update validator configuration.

    Available keys:
    - use_wandb     true/false — enable Weights & Biases run logging
    - wandb_project string    — W&B project name (default: evol-validator)

    Examples:
        evolcli validator config --show
        evolcli validator config --set use_wandb=true
        evolcli validator config --set wandb_project=my-project
    """
    from dotenv import load_dotenv
    load_dotenv()
    config_dir = Path.home() / ".evolai" / "validator"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"

    # Load existing config
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'use_wandb': False,
            'wandb_project': 'evol-validator'
        }

    # Set new value
    if set_key:
        if '=' not in set_key:
            err_console.print("\n❌ Invalid format. Use: --set key=value")
            raise typer.Exit(code=1)

        key, value = set_key.split('=', 1)
        key = key.strip()
        value = value.strip()

        # Parse value
        if key == 'use_wandb':
            config[key] = value.lower() in ['true', '1', 'yes']
        else:
            config[key] = value

        # Save config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        console.print(f"\n[green]✓ Configuration updated:[/green] {key} = {config[key]}\n")

    # Show configuration
    if show or set_key:
        console.print(Panel.fit(
            f"[bold]Validator Configuration[/bold]\n\n"
            f"W&B Logging: [cyan]{config.get('use_wandb', False)}[/cyan]\n"
            f"W&B Project: [cyan]{config.get('wandb_project', 'N/A')}[/cyan]\n\n"
            f"[dim]Config file: {config_file}[/dim]",
            title="Configuration",
            border_style="cyan"
        ))
        console.print()


@validator_app.command("list-local")
def list_local_results(
    track: Optional[str] = typer.Option(
        None,
        "--track",
        help="Filter by track"
    ),
    limit: Optional[int] = typer.Option(
        10,
        "--limit",
        help="Number of results to show"
    )
):
    """
    List locally saved evaluation results.

    Reads JSON files written to ~/.evolai/validator/results/ and
    prints a summary table (timestamp, track, miner count, top scores).

    Examples:
        evolcli validator list-local
        evolcli validator list-local --track transformer --limit 5
    """
    results_dir = Path.home() / ".evolai" / "validator" / "results"
    
    if not results_dir.exists():
        console.print("\n[yellow]No local results found[/yellow]\n")
        return
    
    # Find all result files
    result_files = sorted(results_dir.glob("evaluation_*.json"), reverse=True)
    
    if not result_files:
        console.print("\n[yellow]No evaluation results found[/yellow]\n")
        return
    
    # Filter by track if specified
    if track:
        result_files = [f for f in result_files if track in f.name]
    
    # Limit results
    result_files = result_files[:limit]
    
    console.print(f"\n[bold]Local Evaluation Results[/bold] ([dim]showing {len(result_files)}[/dim])\n")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            console.print(f"[cyan]━[/cyan] {result_file.name}")
            console.print(f"   Track: {data.get('track', 'N/A')} | "
                        f"Miners: {len(data.get('results', []))} | "
                        f"Questions: {data.get('num_questions', 'N/A')} | "
                        f"Timestamp: {data.get('timestamp', 'N/A')[:19]}")
            
            # Show top 3 scores
            results = data.get('results', [])
            if results:
                sorted_results = sorted(results, key=lambda x: x.get('raw_score', 0), reverse=True)
                console.print("   [dim]Top scores:[/dim]", end=" ")
                for i, r in enumerate(sorted_results[:3], 1):
                    console.print(f"UID {r.get('miner_uid')}: {r.get('raw_score', 0):.2f}", end="  ")
                console.print()
            console.print()
            
        except Exception as e:
            console.print(f"[red]Error reading {result_file.name}: {e}[/red]\n")
    
    console.print(f"[dim]Results directory: {results_dir}[/dim]\n")


def _scan_miners_from_chain(
    subtensor,
    netuid: int,
    eval_track: str,
    console,
    verbose: bool = False,
) -> tuple:
    """
    Scan all UIDs in the subnet metagraph for on-chain commitment metadata.

    Returns:
        (miners, uids_without_metadata)
        miners: list of dicts with uid, hotkey, coldkey, model_name, revision, metadata
    """
    from evolai.utils.metadata import decompress_metadata

    metagraph = subtensor.metagraph(netuid)
    miners: list = []
    uids_without_metadata: list = []

    for uid in range(len(metagraph.hotkeys)):
        hotkey = metagraph.hotkeys[uid]
        coldkey = metagraph.coldkeys[uid] if hasattr(metagraph, 'coldkeys') else ""
        try:
            commit_data = subtensor.get_commitment_metadata(netuid, hotkey)
            if not commit_data:
                uids_without_metadata.append(uid)
                continue

            if not (isinstance(commit_data, dict) and 'info' in commit_data):
                uids_without_metadata.append(uid)
                continue

            fields = commit_data['info']['fields']
            if not (fields and len(fields) > 0 and fields[0] and len(fields[0]) > 0):
                uids_without_metadata.append(uid)
                continue

            raw_data = fields[0][0]
            # Field key is RawN where N = byte length (e.g. Raw32, Raw50, Raw64)
            raw_key = next((k for k in raw_data if k.startswith('Raw') and k[3:].isdigit()), None)
            if raw_key is None:
                if verbose:
                    console.print(f"  [yellow]UID {uid}: No RawN field in commitment data[/yellow]")
                uids_without_metadata.append(uid)
                continue

            compressed_bytes = bytes(raw_data[raw_key][0])
            metadata = decompress_metadata(compressed_bytes)
            if not metadata:
                if verbose:
                    console.print(f"  [yellow]UID {uid}: metadata is None after decompression[/yellow]")
                uids_without_metadata.append(uid)
                continue

            if verbose:
                console.print(f"  [dim]UID {uid}: metadata={metadata}[/dim]")

            track_info = metadata.get(eval_track, {})
            if track_info and track_info.get('model_name'):
                miners.append({
                    'uid': uid,
                    'hotkey': hotkey,
                    'coldkey': coldkey,
                    'model_name': track_info['model_name'],
                    'revision': track_info.get('revision', 'main'),
                    'metadata': metadata,
                })
                if verbose:
                    console.print(f"  [green]UID {uid}: Found {eval_track} miner: {track_info['model_name']}[/green]")
            else:
                if verbose:
                    console.print(f"  [yellow]UID {uid}: No {eval_track} track in metadata[/yellow]")
                uids_without_metadata.append(uid)

        except Exception as e:
            if verbose:
                console.print(f"  [red]UID {uid}: Error reading metadata: {e}[/red]")
            uids_without_metadata.append(uid)

    return miners, uids_without_metadata


@validator_app.command("run")
def run_validator(
    wallet_name: Optional[str] = typer.Option(
        None,
        "--wallet",
        help="Validator wallet name"
    ),
    hotkey_name: Optional[str] = typer.Option(
        None,
        "--hotkey",
        help="Validator hotkey name"
    ),
    netuid: Optional[int] = typer.Option(
        47,
        "--netuid",
        help="Subnet UID (default: 47)"
    ),
    num_questions: Optional[int] = typer.Option(
        _NUM_QUESTIONS,
        "--questions",
        help="Number of questions per evaluation"
    ),
    eval_interval: Optional[int] = typer.Option(
        300,
        "--eval-interval",
        help="Seconds between evaluations (default: 300 = 5 min)"
    ),
    weight_interval: Optional[int] = typer.Option(
        1800,
        "--weight-interval",
        help="Seconds between weight updates (default: 1800 = 30 min)"
    ),
    use_wandb: Optional[bool] = typer.Option(
        True,
        "--wandb/--no-wandb",
        help="Enable Weights & Biases logging"
    ),
    wandb_project: Optional[str] = typer.Option(
        "evol-validator",
        "--wandb-project",
        help="W&B project name"
    ),
    debug: Optional[bool] = typer.Option(
        False,
        "--debug",
        help="Debug mode",
        hidden=True,
    ),
    fake_wallet: Optional[bool] = typer.Option(
        False,
        "--fake-wallet",
        help="Testing mode",
        hidden=True,
    ),
    vllm_bin: Optional[str] = typer.Option(
        None,
        "--vllm-bin",
        help="Internal option",
        hidden=True,
    )
):
    """
    Run the validator loop.
    """
    from dotenv import load_dotenv
    load_dotenv()
    import time
    import logging
    import threading
    # Interactive prompts
    if wallet_name is None:
        wallet_name = Prompt.ask("Validator wallet name", default="default")

    if hotkey_name is None:
        hotkey_name = Prompt.ask("Validator hotkey name", default="default")

    # ── Configure logging ────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # Silence noisy third-party loggers
    for _noisy in (
        "httpx", "httpcore", "openai._base_client",
        "urllib3", "asyncio", "bittensor", "websockets",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    if vllm_bin:
        os.environ["VLLM_EXECUTABLE"] = vllm_bin

    console.print("\n[bold cyan]Initializing Validator...[/bold cyan]\n")
    
    if fake_wallet:
        # Generate fake wallet addresses for testing
        import hashlib
        import secrets
        
        # Generate deterministic-looking fake addresses
        fake_seed = secrets.token_hex(16)
        validator_coldkey = "5" + hashlib.sha256(f"cold_{fake_seed}".encode()).hexdigest()[:47]
        validator_hotkey = "5" + hashlib.sha256(f"hot_{fake_seed}".encode()).hexdigest()[:47]
        
        console.print(f"[yellow]⚠ Using FAKE wallet for testing[/yellow]")
        console.print(f"  Fake Coldkey: {validator_coldkey}")
        console.print(f"  Fake Hotkey: {validator_hotkey}\n")
    else:
        with console.status("[cyan]Loading wallet..."):
            try:
                from bittensor_wallet import Wallet
                wallet = Wallet(name=wallet_name, hotkey=hotkey_name)
                
                if not wallet.coldkey_file.exists_on_device():
                    err_console.print(f"\n❌ Coldkey not found for wallet '{wallet_name}'")
                    raise typer.Exit(code=1)
                
                if not wallet.hotkey_file.exists_on_device():
                    err_console.print(f"\n❌ Hotkey '{hotkey_name}' not found")
                    raise typer.Exit(code=1)
                
                validator_coldkey = wallet.coldkeypub.ss58_address
                validator_hotkey = wallet.hotkeypub.ss58_address
                
            except Exception as e:
                err_console.print(f"\n❌ Failed to load wallet: {e}")
                raise typer.Exit(code=1)
        
        console.print(f"[green]✓ Wallet loaded[/green]")
        console.print(f"  Coldkey: {validator_coldkey}")
        console.print(f"  Hotkey: {validator_hotkey}\n")

    # Build validator auth credentials for the proxy challenge endpoint.
    # Fake-wallet path cannot produce real signatures, so auth is skipped
    # (only works when OWNER_API_URL points directly to localhost manager).
    if fake_wallet:
        validator_auth = None
    else:
        from evolai.validator.challenge_client import ValidatorAuth as _ValidatorAuth
        _wallet_ref = wallet  # capture for closure
        validator_auth = _ValidatorAuth(
            hotkey=validator_hotkey,
            sign_fn=lambda msg: _wallet_ref.hotkey.sign(msg.encode()).hex(),
        )
    
    # Connect to Bittensor
    with console.status("[cyan]Connecting to Bittensor..."):
        try:
            import bittensor as bt
            subtensor = bt.Subtensor()
            metagraph = subtensor.metagraph(netuid=netuid)
            console.print(f"[green]✓ Connected to Bittensor[/green] (netuid={netuid}, neurons={len(metagraph.uids)})\n")
        except Exception as e:
            err_console.print(f"\n❌ Failed to connect to Bittensor: {e}")
            raise typer.Exit(code=1)
    
    # ── Transcript logger setup ──────────────────────────────────────────
    # MUST happen AFTER bittensor init: bt.Subtensor() triggers
    # LoggingMachine.before_enable_default() which resets ALL non-primary
    # loggers to CRITICAL.  By configuring here we survive the reset.
    console.print()
    
    # Initialize W&B
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_entity = os.getenv("WANDB_ENTITY", "open-evolai")
            wandb_project = (wandb_project or "evol-validator").strip()
            wandb_api_key = os.getenv("WANDB_API_KEY", "").strip()

            if wandb_api_key:
                wandb.login(key=wandb_api_key, relogin=True)
            else:
                # wandb.login() reads from .netrc, WANDB_API_KEY env, or any
                # active `wandb login` session.  Returns True if credentials
                # were found, False if not.
                if not wandb.login(anonymous="never"):
                    console.print(
                        "[yellow]⚠ W&B login not found, logging disabled[/yellow]"
                    )
                    console.print(
                        f"[dim]Set WANDB_API_KEY or run `wandb login` to log runs to {wandb_entity}/{wandb_project}[/dim]\n"
                    )
                    use_wandb = False

            if use_wandb:
                run_name = f"validator_{validator_coldkey}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

                wandb_run = wandb.init(
                    entity=wandb_entity,
                    project=wandb_project,
                    name=run_name,
                    config={
                        'validator_coldkey': validator_coldkey,
                        'validator_hotkey': validator_hotkey,
                        'netuid': netuid,
                        'tracks': ['transformer', 'mamba2'],
                        'eval_interval': eval_interval,
                        'weight_interval': weight_interval
                    },
                    tags=["validator", f"netuid_{netuid}", "transformer", "mamba2", "loss-eval"]
                )
                console.print(f"[green]✓ W&B initialized[/green] | Run: {run_name}\n")
        except ImportError:
            console.print("[yellow]⚠ wandb not installed, logging disabled[/yellow]")
            console.print("[dim]Install with: pip install wandb[/dim]\n")
            use_wandb = False
        except Exception as e:
            console.print(
                f"[yellow]⚠ W&B initialization failed for {wandb_entity}/{wandb_project}: {e}[/yellow]\n"
            )
            use_wandb = False
    
    # Import evaluation modules
    from evolai.validator.evaluator import ModelValidator, EMAScoreTracker
    from evolai.validator.config import (
        DAILY_ALPHA_EMISSION,
        STAGNATION_BURN_UID,
        EMA_ALPHA,
        EMA_MIN_EVALUATIONS,
        CHALLENGE_WINDOW_SIZE,
        DIRICHLET_BETA,
        REWARD_GAMMA,
        REWARD_MAX,
        REWARD_DECAY,
        OWNER_API_URL,
        get_eval_config_for_model_size,
    )
    from evolai.validator.loss_evaluator import RewardTracker
    from evolai.validator.challenge_client import (
        fetch_challenge,
        fetch_challenge_texts,
        submit_evaluations,
    )

    # Initialize trackers
    ema_tracker = EMAScoreTracker(alpha=EMA_ALPHA)
    reward_tracker = RewardTracker(
        window_size=CHALLENGE_WINDOW_SIZE,
        gamma=REWARD_GAMMA,
        beta=DIRICHLET_BETA,
        reward_max=REWARD_MAX,
        reward_decay=REWARD_DECAY,
    )
    owner_api = OWNER_API_URL

    # ── Submission evaluation tracker ───────────────────────────────────────
    # Tracks which (track, uid, model_name, revision) combos have been
    # successfully evaluated.  A miner is only re-evaluated when it posts a
    # new model or revision on-chain, or when the previous attempt errored.
    _submissions_file = Path.home() / ".evolai" / "validator" / "evaluated_submissions.json"

    def _load_evaluated_submissions() -> dict:
        if _submissions_file.exists():
            try:
                with open(_submissions_file, "r") as _sf:
                    return json.load(_sf)
            except Exception:
                pass
        return {}

    def _save_evaluated_submissions(subs: dict) -> None:
        _submissions_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(_submissions_file, "w") as _sf:
                json.dump(subs, _sf, indent=2)
        except Exception as _se:
            logging.warning(f"[submissions] Failed to persist: {_se}")

    # Structure: {track: {str(uid): {model_name, revision, evaluated_at}}}
    _evaluated_submissions: dict = _load_evaluated_submissions()
    console.print(
        f"[dim]Loaded {sum(len(v) for v in _evaluated_submissions.values())} "
        "prior evaluated submission(s) from disk[/dim]\n"
    )

    active_tracks = ("transformer", "mamba2")

    def _get_track_score_maps(min_evaluations: int) -> Dict[str, Dict[int, float]]:
        all_scores = reward_tracker.get_effective_scores(min_evaluations=min_evaluations)
        track_scores: Dict[str, Dict[int, float]] = {}
        for track_name in active_tracks:
            try:
                track_miners, _ = _scan_miners_from_chain(
                    subtensor, netuid, track_name, console, verbose=False
                )
                track_uids = {miner['uid'] for miner in track_miners}
                track_scores[track_name] = {
                    uid: score for uid, score in all_scores.items() if uid in track_uids
                }
            except Exception as exc:
                logging.warning(
                    f"[weights] Failed to load {track_name} miners for score filtering: {exc}"
                )
                track_scores[track_name] = {}
        return track_scores
    
    console.print(Panel.fit(
        f"[bold]Validator Configuration[/bold]\n\n"
        f"Netuid: [cyan]{netuid}[/cyan]\n"
        f"Tracks: [cyan]transformer, mamba2[/cyan]\n"
        f"Eval Mode: [cyan]Loss-based[/cyan]\n"
        f"Challenge Window: [cyan]{CHALLENGE_WINDOW_SIZE}[/cyan]\n"
        f"Reward Gamma: [cyan]{REWARD_GAMMA}[/cyan]\n"
        f"Reward Decay: [cyan]{REWARD_DECAY}[/cyan]\n"
        f"Eval Interval: [cyan]{eval_interval}s ({eval_interval//60} min)[/cyan]\n"
        f"Weight Interval: [cyan]{weight_interval}s ({weight_interval//60} min)[/cyan]\n"
        f"Weight Mode: [cyan]Proportional-to-Score[/cyan]\n"
        f"W&B Logging: [cyan]{use_wandb}[/cyan]\n"
        f"W&B Project: [cyan]{wandb_project if use_wandb else 'N/A'}[/cyan]\n"
        f"Owner API: [cyan]{owner_api}[/cyan]",
        title="Starting Validator Loop",
        border_style="green"
    ))
    
    if not Confirm.ask("\n[yellow]Start continuous evaluation loop?[/yellow]"):
        console.print("[dim]Cancelled[/dim]")
        raise typer.Exit(code=0)
    
    console.print("\n[bold green]Validator loop started[/bold green] (Press Ctrl+C to stop)\n")
    console.print("━" * 80 + "\n")
    
    # Shared state for weight setting thread
    weight_thread_running = threading.Event()
    weight_thread_running.set()
    weight_thread_stop = threading.Event()
    
    def weight_setting_thread():
        """Background thread for setting weights every 30 minutes"""
        nonlocal last_weight_update

        if fake_wallet:
            console.print("[dim]Weight setting disabled (fake wallet — no signing key)[/dim]\n")
            return

        # Short startup grace period so weight logs don't race with judge-server
        # setup spinners. Weight interval after that is the configured value.
        WEIGHT_STARTUP_DELAY_S = 10
        console.print(f"[dim]Weight setting thread started — first update in {WEIGHT_STARTUP_DELAY_S}s, then every {weight_interval//60}m[/dim]\n")
        if weight_thread_stop.wait(timeout=WEIGHT_STARTUP_DELAY_S):
            return  # stopped before first run

        while weight_thread_running.is_set():
            try:
                
                console.print(f"\n[bold magenta]━━━ Background Weight Update ━━━[/bold magenta]\n")
                
                # Refresh metagraph
                metagraph_local = subtensor.metagraph(netuid=netuid)
                
                # Get effective scores with variance penalties for both tracks
                track_scores = _get_track_score_maps(min_evaluations=EMA_MIN_EVALUATIONS)
                transformer_scores = track_scores["transformer"]
                mamba2_scores = track_scores["mamba2"]
                
                # === Alpha price gating: each track independently targets ~1 TAO/day ===
                alpha_price_tao = 0.0
                per_track_fraction = 1.0
                try:
                    alpha_balance = subtensor.get_subnet_price(netuid=netuid)
                    alpha_price_tao = float(alpha_balance.tao)
                    if alpha_price_tao > 0:
                        per_track_fraction = min(1.0, 1.0 / (alpha_price_tao * DAILY_ALPHA_EMISSION))
                    console.print(f"[cyan]Alpha price: {alpha_price_tao:.6f} TAO | Per-track fraction: {per_track_fraction:.6f} ({per_track_fraction * DAILY_ALPHA_EMISSION:.1f} alpha/day ≈ 1 TAO per track)[/cyan]")
                except Exception as _ape:
                    logging.warning(f"[weights] Failed to fetch alpha price: {_ape}; defaulting per_track_fraction=1.0")
                    console.print(f"  [yellow]⚠ Could not fetch alpha price ({_ape}); using per_track_fraction=1.0[/yellow]")

                # Prepare weights for all neurons
                all_uids = list(range(256))
                weights = [0.0] * 256
                burn_uid_idx = STAGNATION_BURN_UID  # UIDs 0-255 always present in fixed-size list

                console.print(f"[cyan]Each active track: proportional weights, ~1 TAO/day (per_track_fraction={per_track_fraction:.4f})[/cyan]")
                console.print(f"[cyan]Stagnation decay: if same winner for 3+ days, weight decays; remainder burned to UID {STAGNATION_BURN_UID}[/cyan]\n")

                # ── Determine how many tracks are active ─────────────────
                # A track is "active" when it has at least one scored miner.
                # If only one track is active it receives 100% of the emission
                # budget instead of 50%.  If neither track is active every-
                # thing is burned.
                transformer_active = bool(transformer_scores)
                mamba2_active = bool(mamba2_scores)
                active_track_count = int(transformer_active) + int(mamba2_active)

                if active_track_count == 0:
                    console.print(f"  [yellow]⚠ No miners in any track — burning 100% of emissions to UID {STAGNATION_BURN_UID}[/yellow]")
                elif active_track_count == 1:
                    solo_track = "Transformer" if transformer_active else "Mamba2"
                    console.print(f"  [cyan]Only {solo_track} track active — receives 100% of emissions[/cyan]")

                # Per-track share: split equally when both active, else 100%
                track_share = (1.0 / active_track_count) if active_track_count > 0 else 0.0

                # ── Proportional weight distribution per track ────────────
                # For each active track we distribute the track budget
                # proportionally to each miner's effective score rather than
                # giving 100% to the single top miner.  Stagnation decay is
                # applied to the track budget (not per-miner), and the top-UID
                # is still used to update the rank history for that check.
                for _track_name, _track_scores in [
                    ("transformer", transformer_scores),
                    ("mamba2", mamba2_scores),
                ]:
                    if not _track_scores:
                        console.print(f"  {_track_name.capitalize()} track: No miners — emissions redirected")
                        continue

                    _top_uid = max(_track_scores.items(), key=lambda x: x[1])[0]
                    _is_stagnant, _days_unchanged = ema_tracker.check_stagnation(_track_name)
                    _decay = ema_tracker.get_decay_factor(_days_unchanged) if _is_stagnant else 1.0
                    ema_tracker.update_rank_history(_track_name, _top_uid, datetime.utcnow().isoformat())

                    _track_budget = track_share * per_track_fraction * _decay
                    _total_score = sum(_track_scores.values())

                    if _total_score <= 0:
                        console.print(
                            f"  {_track_name.capitalize()} track: all miners have zero score "
                            f"\u2014 emissions redirected"
                        )
                        continue

                    if _is_stagnant:
                        console.print(
                            f"  [yellow]⚠ {_track_name.capitalize()} stagnant "
                            f"{_days_unchanged}d — decay: {_decay:.2f}[/yellow]"
                        )

                    for _uid_w, _score_w in _track_scores.items():
                        if _uid_w in all_uids:
                            _idx_w = all_uids.index(_uid_w)
                            weights[_idx_w] += _track_budget * (_score_w / _total_score)

                    console.print(
                        f"  {_track_name.capitalize()} track: {len(_track_scores)} miners, "
                        f"budget={_track_budget:.6f}, top UID={_top_uid} "
                        f"(score={_track_scores[_top_uid]:.4f})"
                    )

                # ── Burn UID receives all remaining weight ────────────────
                miner_total = sum(weights)
                burn_amount = max(0.0, 1.0 - miner_total)
                if burn_uid_idx is not None:
                    weights[burn_uid_idx] = burn_amount
                console.print(
                    f"  Burn (UID {STAGNATION_BURN_UID}): "
                    f"{burn_amount:.6f} ({burn_amount * 100:.2f}% of emissions)"
                )

                # Log weights being set
                nonzero = [(uid, w) for uid, w in zip(all_uids, weights) if w > 0]
                console.print(f"\n  Setting weights for {len(nonzero)} miners:")
                for _uid, _w in sorted(nonzero, key=lambda x: -x[1]):
                    console.print(f"    UID {_uid}: {_w:.6f}")

                ok, err = subtensor.set_weights(
                    netuid=netuid,
                    wallet=wallet,
                    uids=all_uids,
                    weights=weights,
                    wait_for_inclusion=True,
                    wait_for_finalization=False,
                )

                if ok:
                    console.print(
                        f"  [green]✓ Weights set at "
                        f"{datetime.utcnow().strftime('%H:%M:%S UTC')}[/green]\n"
                    )
                    last_weight_update = datetime.utcnow()
                    if use_wandb and wandb_run:
                        wandb_run.log({
                            'weight_update_success': 1,
                            'transformer_miners_weighted': len(transformer_scores),
                            'mamba2_miners_weighted': len(mamba2_scores),
                            'total_miners_weighted': len(nonzero),
                        })
                        for uid, weight in nonzero:
                            wandb_run.log({f'weights/uid_{uid}': weight})
                else:
                    err_display = err
                    logging.warning(f"[weights] set_weights failed: {err_display}")
                    console.print(f"  [red]✗ Failed to set weights: {err_display}[/red]\n")
                    if use_wandb and wandb_run:
                        wandb_run.log({'weight_update_failure': 1})

            except Exception as e:
                console.print(f"  [red]✗ Weight thread error: {e}[/red]\n")
                if use_wandb and wandb_run:
                    wandb_run.log({'weight_update_error': 1})

            # Wait for next interval (interruptible by stop event)
            if weight_thread_stop.wait(timeout=weight_interval):
                break

        console.print("[dim]Weight setting thread stopped[/dim]\n")

    # Start weight setting background thread
    weight_thread = threading.Thread(target=weight_setting_thread, daemon=True)
    weight_thread.start()

    # Tracking variables
    last_weight_update = datetime.utcnow()
    evaluation_count = 0

    # Single-slot model cache: reuse loaded GPU model across rounds when a
    # miner has not changed their model_name or revision.  Only one model is
    # held in memory at a time; a different (model_name, revision) evicts and
    # replaces the cached entry.
    #   key        – "{model_name}@{revision}" or None
    #   model      – the loaded AutoModelForCausalLM (or None)
    #   tokenizer  – matching tokenizer (or None)
    #   cleanup_fn – the cleanup callable returned by load_model (or None)
    _model_cache: dict = {"key": None, "model": None, "tokenizer": None, "cleanup_fn": None}

    def _evict_model_cache() -> None:
        """Release the cached model from GPU memory and purge its disk cache."""
        import gc
        import torch
        from evolai.validator.evaluator import purge_hf_model_cache as _purge
        cached_key = _model_cache["key"]
        if cached_key is None:
            return
        if _model_cache["cleanup_fn"] is not None:
            try:
                _model_cache["cleanup_fn"]()
            except Exception:
                pass
        _model_cache.update({"key": None, "model": None, "tokenizer": None, "cleanup_fn": None})
        torch.cuda.empty_cache()
        gc.collect()
        # Purge disk: cached_key is "model_name@revision"; strip the revision.
        cached_model_name = cached_key.split("@")[0] if cached_key else None
        if cached_model_name:
            try:
                _purge(cached_model_name)
            except Exception:
                pass
        if cached_key:
            console.print(f"  [dim]Evicted cached model: {cached_key}[/dim]")

    # ── Signal handling (SIGTERM / SIGINT) ──────────────────────────────────
    import signal as _signal
    _stop_requested = threading.Event()

    def _request_stop(signum, frame):  # noqa: ARG001
        console.print(
            f"\n[yellow]Received signal {signum} — stopping after current round[/yellow]"
        )
        _stop_requested.set()

    try:
        _signal.signal(_signal.SIGTERM, _request_stop)
    except (OSError, AttributeError):
        pass  # SIGTERM not reliably available on Windows

    MAX_CRASH_RESTARTS = 10
    CRASH_BACKOFF_BASE_S = 30   # first backoff = 30 s; doubles each crash, capped at 5 min
    _run_start = datetime.utcnow()
    crash_count = 0

    while not _stop_requested.is_set():
      try:
        while not _stop_requested.is_set():
            evaluation_count += 1
            round_start = datetime.utcnow()

            console.print(
                f"[bold cyan]Evaluation Round #{evaluation_count}[/bold cyan] "
                f"— {round_start.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            )

            # ── Refresh metagraph + sync UID ownership ─────────────────────
            try:
                metagraph = subtensor.metagraph(netuid=netuid)
                for _uid_i in range(len(metagraph.hotkeys)):
                    _ck = metagraph.coldkeys[_uid_i] if hasattr(metagraph, 'coldkeys') else ""
                    replaced = reward_tracker.sync_uid(_uid_i, _ck, metagraph.hotkeys[_uid_i])
                    if replaced:
                        console.print(
                            f"  [yellow]⚠ UID {_uid_i}: new miner detected — "
                            f"history wiped[/yellow]"
                        )
                        # Clear submission tracker so the new miner is evaluated fresh.
                        for _trk in active_tracks:
                            _evaluated_submissions.get(_trk, {}).pop(str(_uid_i), None)
                        _save_evaluated_submissions(_evaluated_submissions)
            except Exception as _mg_err:
                logging.warning(f"[run] Metagraph refresh failed: {_mg_err}")
                console.print("  [yellow]⚠ Metagraph refresh failed — using cached state[/yellow]")

            for eval_track in active_tracks:
                console.print(f"[bold]Evaluating {eval_track.upper()} track...[/bold]\n")

                try:
                    # ── Read miners from Bittensor chain ──────────────────────
                    console.print(f"  [cyan]Reading {eval_track} miners from Bittensor chain...[/cyan]")
                    miners, uids_without_meta = _scan_miners_from_chain(
                        subtensor, netuid, eval_track, console, verbose=debug
                    )
                    console.print(
                        f"  Found [bold]{len(miners)}[/bold] {eval_track} miners "
                        f"([dim]{len(uids_without_meta)} UIDs without metadata[/dim])"
                    )

                    if not miners:
                        console.print(
                            f"  [yellow]⚠ No miners for {eval_track} track — skipping[/yellow]\n"
                        )
                        continue

                    round_results = []
                    skipped = []

                    # ── Fetch challenges for ALL miners in parallel ────────────
                    # We fetch first so the challenge hash can be included in the
                    # skip decision below: a miner is only re-evaluated when EITHER
                    # its model/revision changes OR the challenge indices change.
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    import hashlib as _hashlib
                    from evolai.validator.challenge_client import (
                        fetch_challenge as _fetch_challenge,
                        fetch_challenge_texts as _fetch_texts,
                    )

                    def _compute_challenge_hash(datasets: dict) -> str:
                        """Stable 16-char SHA-256 fingerprint of challenge datasets."""
                        _payload = json.dumps(
                            {k: sorted(v) for k, v in sorted(datasets.items())},
                            separators=(',', ':'),
                        )
                        return _hashlib.sha256(_payload.encode()).hexdigest()[:16]

                    console.print(
                        f"  [cyan]Fetching challenges for {len(miners)} miner(s)…[/cyan]"
                    )
                    _prefetched: dict = {}  # uid → (challenge, texts) | None

                    def _fetch_miner_challenge(miner_item):
                        _u = miner_item['uid']
                        try:
                            _ch = _fetch_challenge(_u, owner_api, validator_auth)
                            if _ch is None:
                                return _u, None
                            _tx = _fetch_texts(_ch.datasets)
                            return _u, (_ch, _tx)
                        except Exception as _exc:
                            logging.warning(f"[prefetch] UID {_u}: {_exc}")
                            return _u, None

                    _workers = min(8, max(1, len(miners)))
                    with ThreadPoolExecutor(max_workers=_workers) as _pool:
                        _futs = {_pool.submit(_fetch_miner_challenge, m): m['uid'] for m in miners}
                        for _fut in as_completed(_futs):
                            _uid_r, _res = _fut.result()
                            _prefetched[_uid_r] = _res

                    _prefetch_ok = sum(1 for v in _prefetched.values() if v is not None)
                    console.print(
                        f"  Fetched {_prefetch_ok}/{len(miners)} challenges\n"
                    )

                    # ── Skip miners with unchanged model AND unchanged challenge ──
                    # Re-evaluate only when EITHER the miner posts a new model/
                    # revision on-chain OR the challenge indices rotate.
                    _track_subs = _evaluated_submissions.setdefault(eval_track, {})
                    miners_to_eval: list = []
                    for _m in miners:
                        _prior = _track_subs.get(str(_m['uid']))
                        _rev_norm = _m.get('revision') or 'main'
                        _fetched_item = _prefetched.get(_m['uid'])
                        _ch_hash = (
                            _compute_challenge_hash(_fetched_item[0].datasets)
                            if _fetched_item is not None
                            else None
                        )
                        if (
                            _prior is not None
                            and _prior.get('model_name') == _m['model_name']
                            and _prior.get('revision') == _rev_norm
                            and _ch_hash is not None
                            and _prior.get('challenge_hash') == _ch_hash
                        ):
                            skipped.append({
                                'uid': _m['uid'],
                                'reason': 'already evaluated (no new model or challenge)',
                            })
                        else:
                            miners_to_eval.append(_m)

                    _already_done_count = len(miners) - len(miners_to_eval)
                    if _already_done_count > 0:
                        console.print(
                            f"  [dim]{_already_done_count} miner(s) skipped "
                            f"(no new model or challenge)[/dim]"
                        )
                    if not miners_to_eval:
                        console.print(
                            f"  [yellow]⚠ All {eval_track} miners already evaluated — "
                            f"no new model or challenge detected.[/yellow]\n"
                        )
                        continue

                    # ── Evaluate each miner sequentially ──────────────────────
                    for i, miner in enumerate(miners_to_eval, 1):
                        uid: int        = miner['uid']
                        hotkey: str     = miner['hotkey']
                        coldkey: str    = miner.get('coldkey', '')
                        model_name: str = miner['model_name']
                        revision: str   = miner.get('revision') or 'main'

                        console.print(
                            f"  [{i}/{len(miners_to_eval)}] UID {uid} | "
                            f"{model_name} @ {revision} | "
                            f"hotkey {hotkey[:12]}…"
                        )

                        try:
                            chain_hotkey = metagraph.hotkeys[uid]
                            if chain_hotkey != hotkey:
                                console.print(
                                    f"    [yellow]⚠ UID {uid} hotkey mismatch — "
                                    f"skipping (stale scan)[/yellow]"
                                )
                                skipped.append({'uid': uid, 'reason': 'hotkey mismatch'})
                                continue
                        except IndexError:
                            pass  # uid out of metagraph range

                        _model_obj = _tok = None
                        cleanup_fn = None
                        _using_cache = False
                        _store_in_cache = False
                        cache_key = f"{model_name}@{revision}"
                        try:
                            # ── Model cache check ──────────────────────────────
                            if (
                                _model_cache["key"] == cache_key
                                and _model_cache["model"] is not None
                            ):
                                _model_obj = _model_cache["model"]
                                _tok = _model_cache["tokenizer"]
                                _using_cache = True
                                _store_in_cache = True
                                console.print(
                                    f"    [dim]Reusing loaded model (same revision)[/dim]"
                                )
                            else:
                                # Different model — evict whatever is cached first.
                                if _model_cache["key"] is not None:
                                    _evict_model_cache()

                                # ── Load model via HuggingFace transformers ────
                                model_validator = ModelValidator()
                                with console.status(
                                    f"    [cyan]Loading model (HF transformers)...[/cyan]"
                                ):
                                    _model_obj, _tok, is_vllm, cleanup_fn = model_validator.load_model(
                                        model_name, revision, use_vllm=False
                                    )
                                console.print(f"    Loaded (HF transformers)")
                                # Mark for caching after successful load.
                                _store_in_cache = True

                            # ── Use pre-fetched challenge + texts ─────────────
                            from evolai.validator.loss_evaluator import compute_cross_entropy_loss

                            _fetched = _prefetched.get(uid)
                            if _fetched is None:
                                console.print(
                                    f"    [yellow]⚠ No challenge for UID {uid} — skipping[/yellow]"
                                )
                                skipped.append({'uid': uid, 'reason': 'no challenge'})
                                continue
                            _challenge, _texts = _fetched

                            if not _texts:
                                console.print(
                                    f"    [yellow]⚠ Could not load challenge texts — skipping[/yellow]"
                                )
                                skipped.append({'uid': uid, 'reason': 'texts unavailable'})
                                continue

                            _total_indices = sum(len(v) for v in _challenge.datasets.values())
                            _dataset_summary = ", ".join(
                                f"{n}({len(i)} idx)"
                                for n, i in _challenge.datasets.items()
                            )
                            console.print(f"    Challenge: {_dataset_summary} | Texts: {len(_texts)}")

                            # Detect model change
                            reward_tracker.sync_model(uid, model_name)

                            # ── Pick batch/seqlen for this model's size ────────
                            _num_params_b = (
                                sum(p.numel() for p in _model_obj.parameters()) / 1e9
                            )
                            _eval_batch, _eval_max_seq = get_eval_config_for_model_size(
                                _num_params_b
                            )
                            console.print(
                                f"    Model {_num_params_b:.2f}B → "
                                f"batch={_eval_batch}, seq={_eval_max_seq}"
                            )

                            # ── Compute cross-entropy loss via HF transformer ──
                            import time as _time
                            import torch as _torch
                            from rich.progress import (
                                Progress, BarColumn, TaskProgressColumn,
                                TimeElapsedColumn, TextColumn, MofNCompleteColumn,
                            )
                            _device = "cuda" if _torch.cuda.is_available() else "cpu"
                            _eval_start = _time.monotonic()

                            with Progress(
                                TextColumn("    [cyan]Loss eval[/cyan]"),
                                BarColumn(bar_width=28),
                                MofNCompleteColumn(),
                                TaskProgressColumn(),
                                TimeElapsedColumn(),
                                console=console,
                                transient=False,
                            ) as _prog:
                                _task = _prog.add_task("", total=len(_texts))

                                def _on_progress(done: int, total: int) -> None:
                                    _prog.update(_task, completed=done)

                                _loss = compute_cross_entropy_loss(
                                    _model_obj, _tok, _texts,
                                    batch_size=_eval_batch,
                                    max_length=_eval_max_seq,
                                    device=_device,
                                    progress_callback=_on_progress,
                                )
                                # Ensure bar shows 100% before closing.
                                _prog.update(_task, completed=len(_texts))

                            _eval_elapsed = _time.monotonic() - _eval_start
                            console.print(
                                f"    Loss eval done in [dim]{_eval_elapsed:.1f}s[/dim]"
                            )

                            # ── Record loss, compute reward ────────────────────
                            _dataset_names = list(_challenge.datasets.keys())
                            _reward, _best = reward_tracker.record_loss(
                                uid, _loss, model_name,
                                dataset_name=",".join(_dataset_names),
                                revision=revision,
                            )

                            console.print(
                                f"    Loss [bold]{_loss:.4f}[/bold] | "
                                f"Best [bold]{_best:.4f}[/bold] | "
                                f"Reward [bold]{_reward:.4f}[/bold]"
                            )

                            # ── Mark this submission as successfully evaluated ──
                            # Future rounds skip this (uid, model_name, revision,
                            # challenge_hash) until model or challenge changes.
                            _track_subs[str(uid)] = {
                                'model_name': model_name,
                                'revision': revision,
                                'challenge_hash': _compute_challenge_hash(_challenge.datasets),
                                'evaluated_at': datetime.utcnow().isoformat(),
                            }
                            _save_evaluated_submissions(_evaluated_submissions)

                            round_results.append({
                                'miner_uid': uid,
                                'miner_hotkey': hotkey,
                                'track': eval_track,
                                'model_name': model_name,
                                'revision': revision,
                                'loss': _loss,
                                'best_loss': _best,
                                'reward': _reward,
                                'datasets': {n: len(i) for n, i in _challenge.datasets.items()},
                                'num_indices': _total_indices,
                                'timestamp': round_start.isoformat(),
                            })

                            if use_wandb and wandb_run:
                                wandb_run.log({
                                    f"{eval_track}/miner_{uid}_loss": _loss,
                                    f"{eval_track}/miner_{uid}_best_loss": _best,
                                    f"{eval_track}/miner_{uid}_reward": _reward,
                                    f"{eval_track}/evaluation_count": evaluation_count,
                                })

                        except Exception as _eval_err:
                            import traceback as _tb
                            first_line = (
                                str(_eval_err).splitlines()[0]
                                if str(_eval_err)
                                else repr(_eval_err)
                            )
                            console.print(f"    [red]✗ Failed: {first_line}[/red]")
                            console.print(
                                f"      [dim]{_tb.format_exc().splitlines()[-1]}[/dim]"
                            )
                            skipped.append({'uid': uid, 'reason': f'error: {first_line[:60]}'})
                            if use_wandb and wandb_run:
                                wandb_run.log({f"{eval_track}/miner_{uid}_error": 1})
                            # Do not keep a model in cache after an error —
                            # GPU state may be compromised (e.g. after OOM).
                            _store_in_cache = False
                        finally:
                            import gc as _gc_cleanup
                            import torch as _torch_cleanup

                            if _store_in_cache and not _using_cache:
                                # Freshly loaded, evaluation succeeded (or was a clean
                                # skip) — store model in the cache for next round.
                                _model_cache["key"] = cache_key
                                _model_cache["model"] = _model_obj
                                _model_cache["tokenizer"] = _tok
                                _model_cache["cleanup_fn"] = cleanup_fn
                                cleanup_fn = None   # ownership transferred to cache
                                _model_obj = _tok = None
                                _gc_cleanup.collect()

                            elif not _store_in_cache:
                                # Error path — evict any cached entry for this key
                                # so the next round reloads from scratch.
                                if _model_cache["key"] == cache_key:
                                    _evict_model_cache()
                                # Drop local refs before cache/empty_cache.
                                _model_obj = _tok = None
                                if cleanup_fn is not None:
                                    try:
                                        cleanup_fn()
                                    except Exception:
                                        pass
                                    cleanup_fn = None
                                _torch_cleanup.cuda.empty_cache()
                                _gc_cleanup.collect()
                                try:
                                    from evolai.validator.evaluator import purge_hf_model_cache
                                    purge_hf_model_cache(model_name)
                                except Exception:
                                    pass

                            else:
                                # Cache-hit path (normal or skip) — model stays in
                                # cache, just drop local refs and do light GC.
                                _model_obj = _tok = None
                                _gc_cleanup.collect()

                    # ── Save round results ─────────────────────────────────────
                    if round_results:
                        results_dir = Path.home() / ".evolai" / "validator" / "results"
                        results_dir.mkdir(parents=True, exist_ok=True)
                        results_file = results_dir / (
                            f"evaluation_{eval_track}_"
                            f"{round_start.strftime('%Y%m%d_%H%M%S')}.json"
                        )
                        with open(results_file, 'w') as _f:
                            json.dump({
                                'track': eval_track,
                                'mode': 'loss-based',
                                'timestamp': round_start.isoformat(),
                                'results': round_results,
                                'skipped': skipped,
                            }, _f, indent=2)
                        console.print(
                            f"\n  [green]✓ {eval_track.upper()}:[/green] "
                            f"{len(round_results)} evaluated, "
                            f"{len(skipped)} skipped — {results_file.name}"
                        )
                        if use_wandb and wandb_run:
                            wandb_run.log({
                                f"{eval_track}/miners_evaluated": len(round_results),
                                f"{eval_track}/miners_skipped": len(skipped),
                            })

                        owner_results = []
                        for _result in round_results:
                            _state = reward_tracker.get_miner_state(_result['miner_uid'])
                            owner_results.append({
                                'miner_uid': _result['miner_uid'],
                                'miner_hotkey': _result['miner_hotkey'],
                                'raw_score': _result['reward'],
                                'effective_score': _state.cumulative_reward if _state else _result['reward'],
                                'dataset_distribution': {
                                    'datasets': _result['datasets'],
                                    'track': _result['track'],
                                    'model_name': _result['model_name'],
                                    'revision': _result['revision'],
                                    'loss': _result['loss'],
                                    'best_loss': _result['best_loss'],
                                    'reward': _result['reward'],
                                },
                            })

                        submitted = submit_evaluations(
                            evaluation_round=evaluation_count,
                            judge_model=f"loss-based-hf-transformers:{eval_track}",
                            results=owner_results,
                            owner_api_url=owner_api,
                            auth=validator_auth,
                        )
                        if submitted:
                            console.print(
                                f"  [green]✓ Submitted {len(owner_results)} {eval_track} results to owner proxy[/green]"
                            )
                            if use_wandb and wandb_run:
                                wandb_run.log({f"{eval_track}/owner_submit_success": 1})
                        else:
                            console.print(
                                f"  [yellow]⚠ Failed to submit {eval_track} results to owner proxy[/yellow]"
                            )
                            if use_wandb and wandb_run:
                                wandb_run.log({f"{eval_track}/owner_submit_failure": 1})
                    else:
                        console.print(f"  [yellow]No {eval_track} results this round.[/yellow]")

                except Exception as _track_err:
                    console.print(f"  [red]✗ Track {eval_track} failed: {_track_err}[/red]\n")
                    if use_wandb and wandb_run:
                        wandb_run.log({f"{eval_track}/track_error": 1})

            # ── Loss-based leaderboard ─────────────────────────────────────
            console.print("[bold]Current Leaderboard (loss-based):[/bold]\n")
            _leaderboards = _get_track_score_maps(min_evaluations=1)
            for _tname in active_tracks:
                _lb_scores = _leaderboards[_tname]
                if not _lb_scores:
                    console.print(f"  [dim]{_tname.upper()}: no miners scored yet.[/dim]")
                    continue
                console.print(f"[bold]{_tname.upper()}[/bold]")
                _sorted = sorted(_lb_scores.items(), key=lambda x: x[1], reverse=True)
                _lb_table = Table(show_header=True, header_style="bold cyan")
                _lb_table.add_column("Rank", justify="right", style="dim")
                _lb_table.add_column("UID", justify="right")
                _lb_table.add_column("Score", justify="right", style="bold")
                _lb_table.add_column("Best Loss", justify="right")
                _lb_table.add_column("Evals", justify="right")
                for _rank, (_uid, _sc) in enumerate(_sorted[:15], 1):
                    _st = reward_tracker.get_miner_state(_uid)
                    _bl = f"{_st.best_loss:.4f}" if _st and _st.best_loss < float('inf') else "N/A"
                    _ec = str(_st.eval_count) if _st else "0"
                    _lb_table.add_row(str(_rank), str(_uid), f"{_sc:.4f}", _bl, _ec)
                console.print(_lb_table)
                console.print()

            # ── Wait until next round (interruptible by stop signal) ──────────
            round_duration = (datetime.utcnow() - round_start).total_seconds()
            wait_time = max(0, eval_interval - round_duration)
            if wait_time > 0:
                console.print(f"[dim]Next evaluation in {int(wait_time)}s…[/dim]\n")
            console.print("━" * 80 + "\n")
            if wait_time > 0:
                _stop_requested.wait(timeout=wait_time)

      except KeyboardInterrupt:
        _stop_requested.set()

      except Exception as _crash_err:
        import traceback as _tb
        crash_count += 1
        logging.error(
            f"[run] Validator loop crash #{crash_count}: {_crash_err}",
            exc_info=True,
        )
        console.print(
            f"\n[red]Validator crashed (#{crash_count}/{MAX_CRASH_RESTARTS}): "
            f"{_crash_err}[/red]"
        )
        console.print(f"[dim]{_tb.format_exc().splitlines()[-1]}[/dim]")

        # Always clean up GPU state before restart — OOM can corrupt the cache.
        _evict_model_cache()

        if crash_count >= MAX_CRASH_RESTARTS:
            console.print(
                f"[red]Exceeded {MAX_CRASH_RESTARTS} consecutive crashes — "
                "exiting.[/red]"
            )
            _stop_requested.set()
            break

        backoff = min(300, CRASH_BACKOFF_BASE_S * crash_count)
        console.print(
            f"[yellow]Restarting validator loop in {backoff}s "
            f"(crash #{crash_count})…[/yellow]\n"
        )
        if use_wandb and wandb_run:
            try:
                wandb_run.log({"validator_crash": 1, "crash_count": crash_count})
            except Exception:
                pass
        if _stop_requested.wait(timeout=backoff):
            break  # stop was requested during backoff
        # Successful restart — decay crash count so brief hiccups don't ban us
        crash_count = max(0, crash_count - 1)
        continue  # restart inner loop

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    console.print("\n[yellow]Validator stopped[/yellow]")

    weight_thread_running.clear()
    weight_thread_stop.set()
    weight_thread.join(timeout=5)

    _evict_model_cache()

    if use_wandb and wandb_run:
        wandb_run.finish()

    total_runtime_h = (datetime.utcnow() - _run_start).total_seconds() / 3600
    console.print(f"\n[bold]Session Summary:[/bold]")
    console.print(f"  Total Rounds: {evaluation_count}")
    console.print(f"  Total Runtime: {total_runtime_h:.2f} hours")
    console.print()


if __name__ == "__main__":
    validator_app()
