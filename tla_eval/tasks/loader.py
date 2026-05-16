"""
Task loader for benchmark test cases.

This module handles loading source code from GitHub repositories
and preparing them as generation tasks with appropriate prompts.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Dict, List
import yaml
from ..methods.base import GenerationTask


class TaskLoader:
    """Loads benchmark tasks by cloning repositories and extracting source code."""
    
    def __init__(self, tasks_dir: str = "tla_eval/tasks", cache_dir: str = "data/repositories"):
        """
        Initialize task loader.
        
        Args:
            tasks_dir: Directory containing task definitions
            cache_dir: Directory to cache cloned repositories
        """
        self.tasks_dir = Path(tasks_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_task(self, task_name: str, source_file: str | List[str] | None = None,
                  traces_folder: str = None) -> GenerationTask:
        """
        Load a specific task by name, automatically cloning repository if needed.

        Args:
            task_name: Name of the task (e.g., "etcd")
            source_file: Specific source file path, or None for default
            traces_folder: Path to the folder containing traces, or None if not available

        Returns:
            GenerationTask instance with source code and appropriate prompt
        """
        task_dir = self.tasks_dir / task_name
        
        if not task_dir.exists():
            available = self.list_available_tasks()
            raise ValueError(f"Task '{task_name}' not found. Available: {available}")
        
        # Load task metadata
        metadata_file = task_dir / "task.yaml"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Task metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        
        # Build list of available source entries (path + optional description)
        source_entries = self._build_source_entries(metadata)

        # Determine which source file(s) to use (support single path or list)
        selected_entries = self._select_source_entries(metadata, source_entries, source_file)

        # Clone repository and get source code (concatenate if multiple)
        source_code = self._get_source_code(metadata['repository'], selected_entries)

        return GenerationTask(
            source_code=source_code,
            task_name=task_name,
            system_type=metadata['system_type'],
            language=metadata['language'],
            description=metadata['description'],
            spec_module=metadata.get('specModule', task_name),
            extra_info={
                'file_path': [e['path'] for e in selected_entries] if len(selected_entries) > 1 else selected_entries[0]['path'],
                'focus': [e.get('description', '') for e in selected_entries] if len(selected_entries) > 1 else selected_entries[0].get('description', ''),
                'repository_url': metadata['repository']['url'],
                'traces_folder': traces_folder or metadata.get('traces_folder'),
            }
        )
    
    def _build_source_entries(self, metadata: Dict) -> List[Dict]:
        """
        Normalize source entries from metadata.

        Supports legacy 'source_files' list and new multi-entry 'default_source_file'.
        """
        entries: List[Dict] = []

        # Prefer explicit source_files list if present (legacy format)
        if metadata.get('source_files'):
            for file_info in metadata['source_files']:
                entries.append({
                    'path': file_info['path'],
                    'description': file_info.get('description', '')
                })
            return entries

        # Otherwise, infer from default_source_file which may be a string or list
        default_src = metadata.get('default_source_file')
        if isinstance(default_src, list):
            for item in default_src:
                if isinstance(item, dict):
                    entries.append({
                        'path': item['path'],
                        'description': item.get('description', '')
                    })
                else:
                    entries.append({'path': item, 'description': ''})
        elif isinstance(default_src, str):
            entries.append({'path': default_src, 'description': ''})

        return entries

    def _select_source_entries(self, metadata: Dict, entries: List[Dict], source_file: str | List[str] | None) -> List[Dict]:
        """
        Choose one or more source entries based on caller input and metadata defaults.
        """
        if not entries:
            raise ValueError("No source files defined in task metadata")

        default_src = metadata.get('default_source_file')

        # If caller didn't specify and default is a list, use all defaults
        if source_file is None and isinstance(default_src, list):
            return entries

        # Normalize caller input to list of paths
        if source_file is None:
            source_paths = [default_src]
        elif isinstance(source_file, list):
            source_paths = source_file
        else:
            source_paths = [source_file]

        selected = []
        missing = []
        for path in source_paths:
            match = next((e for e in entries if e['path'] == path), None)
            if match:
                selected.append(match)
            else:
                missing.append(path)

        if missing:
            available = [e['path'] for e in entries]
            raise ValueError(f"Source file(s) {missing} not found. Available: {available}")

        return selected

    def _get_source_code(self, repo_info: Dict, file_entries: List[Dict]) -> str:
        """
        Clone repository if needed and extract source code.
        
        Args:
            repo_info: Repository information from task.yaml
            file_entries: List of file entries with 'path' (and optional 'description')
            
        Returns:
            Source code content
        """
        repo_url = repo_info['url']
        branch = repo_info.get('branch', 'main')
        
        # Create repository cache directory name from URL
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_cache_dir = self.cache_dir / repo_name
        
        # Clone repository if not already cached
        if not repo_cache_dir.exists():
            print(f"Cloning repository: {repo_url}")
            try:
                # Clone with specific commit if specified
                if 'commit' in repo_info:
                    commit = repo_info['commit']
                    print(f"Using fixed commit: {commit}")
                    subprocess.run([
                        'git', 'clone', repo_url, str(repo_cache_dir)
                    ], check=True, capture_output=True, text=True)
                    subprocess.run([
                        'git', 'checkout', commit
                    ], cwd=repo_cache_dir, check=True, capture_output=True, text=True)
                else:
                    subprocess.run([
                        'git', 'clone', '--depth', '1', 
                        '--branch', branch, 
                        repo_url, str(repo_cache_dir)
                    ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone repository {repo_url}: {e.stderr}")
        
        # Read and concatenate source files
        contents = []
        for entry in file_entries:
            file_path = entry['path']
            source_file_path = repo_cache_dir / file_path
            if not source_file_path.exists():
                raise FileNotFoundError(f"Source file not found in repository: {file_path}")

            with open(source_file_path, 'r', encoding='utf-8') as f:
                file_body = f.read()

            desc = entry.get('description', '')
            header = [
                f"===== FILE: {file_path} =====",
                f"DESCRIPTION: {desc}" if desc else ""
            ]
            header_text = "\n".join([h for h in header if h]) + "\n"
            contents.append(header_text + file_body)

        return "\n\n".join(contents)
        
    def get_task_prompt(self, task_name: str, method_name: str,
                        language: str = "TLA+") -> str:
        """
        Read a prompt template, with per-language overrides.

        Resolution order:
          1. tasks/<task>/prompts/<lang>/<method>.txt
          2. tasks/<task>/prompts/<method>_<lang>.txt
          3. tasks/<task>/prompts/<method>.txt
             — only consulted when `language` is TLA+ (or unspecified). For
             non-TLA+ languages, missing prompts are a hard error; we never
             silently fall back to a TLA+ prompt because that would send the
             model the wrong instructions and silently generate TLA+ output
             under a `--language SAM` (etc.) banner.

        `<lang>` is normalized to lowercase with '+' stripped (e.g. "TLA+"
        -> "tla", "Alloy" -> "alloy").
        """
        lang_key = language.lower().replace("+", "").strip() if language else ""
        prompts_dir = self.tasks_dir / task_name / "prompts"
        is_tla = (not lang_key) or lang_key in ("tla", "tla_plus", "tlaplus")

        candidates: List[Path] = []
        if not is_tla:
            candidates.append(prompts_dir / lang_key / f"{method_name}.txt")
            candidates.append(prompts_dir / f"{method_name}_{lang_key}.txt")
        else:
            candidates.append(prompts_dir / f"{method_name}.txt")

        for path in candidates:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()

        raise FileNotFoundError(
            f"Prompt file not found for task='{task_name}', method='{method_name}', "
            f"language='{language}'. Tried: " + ", ".join(str(p) for p in candidates) +
            (". Non-TLA+ languages do not fall back to the TLA+ prompt; add a "
             f"per-language prompt under prompts/{lang_key}/ to fix." if not is_tla else "")
        )
    
    def list_available_tasks(self) -> List[str]:
        """List all available task names."""
        if not self.tasks_dir.exists():
            return []
        
        tasks = []
        for item in self.tasks_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it has task.yaml
                if (item / "task.yaml").exists():
                    tasks.append(item.name)
        
        return sorted(tasks)
    
    def list_task_source_files(self, task_name: str) -> List[Dict]:
        """List all available source files for a task."""
        task_dir = self.tasks_dir / task_name
        metadata_file = task_dir / "task.yaml"
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        
        return metadata['source_files']
    
    def get_task_info(self, task_name: str) -> Dict:
        """Get metadata about a specific task."""
        task_dir = self.tasks_dir / task_name
        metadata_file = task_dir / "task.yaml"
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def clear_cache(self, task_name: str = None):
        """
        Clear repository cache.
        
        Args:
            task_name: Clear cache for specific task, or None for all
        """
        if task_name is None:
            # Clear all cache
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Clear cache for specific task
            task_info = self.get_task_info(task_name)
            repo_url = task_info['repository']['url']
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            repo_cache_dir = self.cache_dir / repo_name
            
            if repo_cache_dir.exists():
                shutil.rmtree(repo_cache_dir)


# Global task loader instance
_task_loader = None

def get_task_loader() -> TaskLoader:
    """Get global task loader instance."""
    global _task_loader
    if _task_loader is None:
        _task_loader = TaskLoader()
    return _task_loader

def load_task(task_name: str, source_file: str = None, traces_folder: str = None) -> GenerationTask:
    """Convenience function to load a task."""
    return get_task_loader().load_task(task_name, source_file, traces_folder)
